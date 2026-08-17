/**
 * physx_bindings.cpp — pybind11 wrapper for PhysX 5 SDK (T2: full API).
 *
 * PhysX 5 Reduced-Coordinate Articulation API (per-axis):
 *   - Joint methods take PxArticulationAxis::Enum (eTWIST for revolute)
 *   - setJointPosition(axis, val), getJointPosition(axis)
 *   - setJointVelocity(axis, val), getJointVelocity(axis)
 *   - setDriveTarget(axis, target), setDriveParams(axis, PxArticulationDrive)
 *   - setLimitParams(axis, PxArticulationLimit)
 *   - setFrictionCoefficient(val) [deprecated], setFrictionParams(axis, params)
 *   - Scene::addArticulation(art), NOT addActor
 *
 * Quaternion convention: PxQuat(x,y,z,w) → numpy [w,x,y,z]
 */
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include <PxPhysicsAPI.h>
#include <foundation/PxFoundation.h>
#include <algorithm>
#include <map>
#include <mutex>
#include <vector>
#include <string>

namespace py = pybind11;
using namespace physx;

struct Articulation;
// Opt-in contact capture (PHYSX_CONTACT_DEBUG=1).  When off, the callback is
// a no-op and no contact points are stored — training runs leak-free.
static bool g_contact_debug = getenv("PHYSX_CONTACT_DEBUG") != nullptr;

// ═══════════════════════════════════════════════════════════════════════
struct MinimalErrorCallback : PxErrorCallback {
    void reportError(PxErrorCode::Enum code, const char *message,
                     const char *file, int line) override {
        if (code == PxErrorCode::eDEBUG_INFO || code == PxErrorCode::eDEBUG_WARNING) return;
        fprintf(stderr, "PhysX [%d]: %s (%s:%d)\n", code, message, file, line);
    }
};

static PxFilterFlags filter_shader(
    PxFilterObjectAttributes a0, PxFilterData d0,
    PxFilterObjectAttributes a1, PxFilterData d1,
    PxPairFlags &pf, const void *cb, PxU32 cbs) {
    (void)d0;(void)d1;(void)cb;(void)cbs;
    // Kill articulation-articulation pairs (self-collision): with corrected
    // link frames the per-link fallback capsules overlap at the joints, and
    // self-collision would explode the solver.  Only robot-vs-ground matters
    // for this task.  (The permissive shader was fine while the broken frames
    // kept the shapes far apart.)
    auto t0 = PxGetFilterObjectType(a0), t1 = PxGetFilterObjectType(a1);
    if (t0 == PxFilterObjectType::eARTICULATION &&
        t1 == PxFilterObjectType::eARTICULATION)
        return PxFilterFlag::eKILL;
    pf = PxPairFlag::eCONTACT_DEFAULT | PxPairFlag::eNOTIFY_TOUCH_FOUND
       | PxPairFlag::eNOTIFY_TOUCH_LOST;
    if (g_contact_debug)
        pf |= PxPairFlag::eNOTIFY_CONTACT_POINTS;
    return PxFilterFlag::eDEFAULT;
}

// ═══════════════════════════════════════════════════════════════════════
// Globals
// ═══════════════════════════════════════════════════════════════════════
static PxDefaultAllocator   *g_allocator  = nullptr;
static MinimalErrorCallback *g_error_cb   = nullptr;
static PxFoundation         *g_foundation = nullptr;
static PxPhysics            *g_physics    = nullptr;
static std::vector<PxScene*> g_scenes;

// ═══════════════════════════════════════════════════════════════════════
// Helpers — quat conversion: PxQuat(x,y,z,w) ↔ numpy[w,x,y,z]
// ═══════════════════════════════════════════════════════════════════════
static py::array_t<float> quat_to_np(const PxQuat &q) {
    float b[4]={q.w,q.x,q.y,q.z}; return py::array_t<float>({4},b);
}
static py::array_t<float> v3_to_np(const PxVec3 &v) {
    float b[3]={v.x,v.y,v.z}; return py::array_t<float>({3},b);
}
static PxVec3 np_to_v3(const py::array_t<float> &a) {
    auto b=a.unchecked<1>(); return PxVec3(b(0),b(1),b(2));
}
static PxQuat np_to_quat(const py::array_t<float> &a) {
    auto b=a.unchecked<1>(); return PxQuat(b(1),b(2),b(3),b(0));
}
static PxTransform np_to_xf(const py::array_t<float> &p,const py::array_t<float> &q) {
    return PxTransform(np_to_v3(p), np_to_quat(q));
}

// ═══════════════════════════════════════════════════════════════════════
// ═══════════════════════════════════════════════════════════════════════
// Debug contact capture — records contact points with actor pointers so
// Python can identify WHICH links/shapes are touching (foot-contact audit).
// ═══════════════════════════════════════════════════════════════════════
struct DebugContact {
    PxVec3 pa, pb;        // world positions of the two actors
    const void *actA, *actB;
    float sep;            // contact point separation
};
static std::vector<DebugContact> g_debug_contacts;
static std::mutex g_debug_mutex;  // std::mutex: PxMutex needs the foundation, which isn't up yet at static-init

struct DebugCallback : PxSimulationEventCallback {
    void onConstraintBreak(PxConstraintInfo*, PxU32) override {}
    void onWake(PxActor**, PxU32) override {}
    void onSleep(PxActor**, PxU32) override {}
    void onTrigger(PxTriggerPair*, PxU32) override {}
    void onAdvance(const PxRigidBody* const*, const PxTransform*, const PxU32) override {}
    void onContact(const PxContactPairHeader &h, const PxContactPair *pairs, PxU32 n) override {
        if(!g_contact_debug) return;
        auto *a0 = static_cast<PxRigidActor*>(h.actors[0]);
        auto *a1 = static_cast<PxRigidActor*>(h.actors[1]);
        for(PxU32 i=0;i<n;i++){
            PxContactPairPoint pts[8];
            PxU32 m = pairs[i].extractContacts(pts, 8);
            for(PxU32 j=0;j<m;j++){
                g_debug_mutex.lock();
                g_debug_contacts.push_back({a0->getGlobalPose().p,
                                            a1->getGlobalPose().p,
                                            h.actors[0], h.actors[1], pts[j].separation});
                g_debug_mutex.unlock();
            }
        }
    }
};
static DebugCallback g_debug_cb;

// Scene
// ═══════════════════════════════════════════════════════════════════════
struct Scene {
    PxScene *ptr = nullptr;
    int pos_iters, vel_iters;
    std::vector<PxMaterial*> materials;
    std::vector<PxRigidStatic*> statics;
    std::vector<Articulation*> articulations;

    ~Scene();
    Scene(const Scene&)=delete; Scene& operator=(const Scene&)=delete;

    void simulate(float dt)          { ptr->simulate(dt); }
    void fetch_results(bool b=true) { ptr->fetchResults(b); }

    int create_material(float sf, float df, float r) {
        auto *m = g_physics->createMaterial(sf, df, r);
        materials.push_back(m); return (int)materials.size()-1;
    }
    void add_ground_plane(int midx, py::array_t<float> normal);
    void add_articulation(Articulation *art);

    py::list get_contacts() {
        py::list out;
        g_debug_mutex.lock();
        for(auto &c : g_debug_contacts)
            out.append(py::make_tuple(c.pa.x, c.pa.y, c.pa.z, c.pb.x, c.pb.y, c.pb.z,
                                      (py::ssize_t)c.actA, (py::ssize_t)c.actB, c.sep));
        g_debug_mutex.unlock();
        return out;
    }
    void clear_contacts() {
        g_debug_mutex.lock(); g_debug_contacts.clear(); g_debug_mutex.unlock();
    }
};

// ═══════════════════════════════════════════════════════════════════════
// Articulation
// ═══════════════════════════════════════════════════════════════════════
struct Articulation {
    PxArticulationReducedCoordinate *ptr = nullptr;
    PxScene *scene_ptr = nullptr;

    // Links in creation order (link[0] = root)
    std::vector<PxArticulationLink*>                links;
    // Joints in DOF order: joints[i] = inbound joint of the non-root link
    std::vector<PxArticulationJointReducedCoordinate*> joints;
    std::vector<PxArticulationAxis::Enum>              joint_axes;  // per-joint axis
    std::vector<std::string>   link_names;
    std::map<std::string,int>  name_to_idx;
    struct DriveCfg { float kp,kd,force; };
    std::vector<DriveCfg> drive_cfgs;

    ~Articulation();
    Articulation(const Articulation&)=delete; Articulation& operator=(const Articulation&)=delete;

    int  add_link(int parent_idx, const std::string &name);
    void add_joint(int parent_idx, int child_idx, int axis,
                   float kp, float kd, float force_limit);
    void finalize(py::array_t<float> mass_arr, py::array_t<float> inert_arr,
                  py::array_t<float> pos_arr, py::array_t<float> quat_arr,
                  py::array_t<int> parent_arr,
                  py::array_t<float> com_pos, py::array_t<float> com_quat,
                  py::array_t<int> joint_axis_arr,
                  py::array_t<float> joint_lower, py::array_t<float> joint_upper,
                  py::array_t<float> joint_friction,
                  int pos_iters, int vel_iters,
                  const std::string& drive_type_str,
                  bool local_poses);

    // State
    py::array_t<float> get_joint_positions();
    py::array_t<float> get_joint_velocities();
    std::pair<py::array_t<float>,py::array_t<float>> get_root_world_pose();
    std::pair<py::array_t<float>,py::array_t<float>> get_root_world_velocity();
    std::pair<py::array_t<float>,py::array_t<float>> get_link_world_pose(int idx);
    int  get_link_index(const std::string &name) { return name_to_idx.at(name); }
    py::ssize_t get_link_actor_ptr(int lidx) { return (py::ssize_t)links[lidx]; }
    int  num_joints() const { return (int)joints.size(); }
    int  num_links()  const { return (int)links.size(); }

    void set_root_world_pose(py::array_t<float> p, py::array_t<float> q);
    void set_root_world_velocity(py::array_t<float> lin, py::array_t<float> ang);
    void set_joint_positions(py::array_t<float> qpos);
    void set_joint_velocities(py::array_t<float> qvel);
    void set_joint_drive_targets(py::array_t<float> targets);
    void set_joint_drive_params(int idx, float kp, float kd, float force_limit,
                                const std::string& drive_type_str);
    void set_solver_iterations(int pi, int vi);

    // Isaac alignment (batch 3): per-joint armature (reflected rotor inertia,
    // added to joint-space inertia) and per-axis max joint velocity (Isaac
    // velocity_limit_sim), plus per-link depenetration clamp (Isaac = 1.0).
    void set_joint_armatures(py::array_t<float> armatures);
    void set_joint_velocity_limits(py::array_t<float> vlims);
    void set_max_depenetration_velocity(float v);

    // Joint frame correction (T3: fix parentPose/childPose after createLink)
    void set_joint_parent_pose(int joint_idx, py::array_t<float> pos, py::array_t<float> quat);
    void set_joint_child_pose(int joint_idx, py::array_t<float> pos, py::array_t<float> quat);

    // Route B: manual per-substep PD (bypass articulation drive)
    // Applies world-space torque to link[link_idx] via PxRigidBody::addTorque
    void add_link_torque(int link_idx, float x, float y, float z);

    // Shapes
    void attach_sphere(int lidx, float r, py::array_t<float> p, py::array_t<float> q);
    void attach_box(int lidx, float hx,float hy,float hz, py::array_t<float> p, py::array_t<float> q);
    void attach_capsule(int lidx, float r, float hh, py::array_t<float> p, py::array_t<float> q);

    // Shared material (created once per articulation)
    PxMaterial *shape_mat = nullptr;
    PxMaterial* get_mat() {
        if(!shape_mat) shape_mat = g_physics->createMaterial(0.6f, 0.5f, 0.0f);
        return shape_mat;
    }
};

// ═══════════════════════════════════════════════════════════════════════
// Scene impl
// ═══════════════════════════════════════════════════════════════════════
Scene::~Scene() {
    if (ptr) {
        // Only release if the scene is still in g_scenes (not already freed by release_foundation).
        auto it=std::find(g_scenes.begin(),g_scenes.end(),ptr);
        if(it!=g_scenes.end()) {
            ptr->release();
            g_scenes.erase(it);
        }
        ptr=nullptr;
    }
}
void Scene::add_ground_plane(int midx, py::array_t<float> n_arr) {
    PxVec3 n = np_to_v3(n_arr).getNormalized();
    PxPlane plane(n, 0.0f);
    PxRigidStatic *g = PxCreatePlane(*g_physics, plane, *materials[midx]);
    ptr->addActor(*g); statics.push_back(g);
}
void Scene::add_articulation(Articulation *art) {
    if(!art->ptr) throw std::runtime_error("articulation not finalized");
    ptr->addArticulation(*art->ptr);
    art->scene_ptr = ptr;
    articulations.push_back(art);
}

// ═══════════════════════════════════════════════════════════════════════
// Articulation impl
// ═══════════════════════════════════════════════════════════════════════
Articulation::~Articulation() {
    // If owned by a scene, the scene releases the PhysX object.
    // Otherwise release it directly.
    if(ptr && !scene_ptr) { ptr->release(); }
    ptr=nullptr;
}

int Articulation::add_link(int parent_idx, const std::string &name) {
    if(ptr) throw std::runtime_error("already finalized");
    int idx=(int)links.size();
    link_names.push_back(name);
    name_to_idx[name]=idx;
    links.push_back(nullptr);  // placeholder
    return idx;
}

void Articulation::add_joint(int parent_idx, int child_idx, int axis,
                              float kp, float kd, float force_limit) {
    if(ptr) throw std::runtime_error("already finalized");
    int expected_child = (int)joints.size() + 1;
    if(child_idx != expected_child)
        throw std::runtime_error("add_joint: expected child_idx=" +
            std::to_string(expected_child)+", got "+std::to_string(child_idx));
    if(parent_idx < 0 || parent_idx >= child_idx)
        throw std::runtime_error("add_joint: invalid parent_idx");
    PxArticulationAxis::Enum ax;
    if(axis==0) ax=PxArticulationAxis::eTWIST;
    else if(axis==1) ax=PxArticulationAxis::eSWING1;
    else if(axis==2) ax=PxArticulationAxis::eSWING2;
    else throw std::runtime_error("add_joint: invalid axis (0=X,1=Y,2=Z)");
    joints.push_back(nullptr);
    joint_axes.push_back(ax);
    drive_cfgs.push_back({kp, kd, force_limit});
}

// Parse drive type string → enum
static PxArticulationDriveType::Enum parse_drive_type(const std::string& s) {
    if(s=="FORCE") return PxArticulationDriveType::eFORCE;
    return PxArticulationDriveType::eACCELERATION; // default
}

void Articulation::finalize(
    py::array_t<float> mass_arr, py::array_t<float> inert_arr,
    py::array_t<float> pos_arr, py::array_t<float> quat_arr,
    py::array_t<int> parent_arr,
    py::array_t<float> com_pos, py::array_t<float> com_quat,
    py::array_t<int> axis_arr,
    py::array_t<float> lower_arr, py::array_t<float> upper_arr,
    py::array_t<float> fric_arr,
    int pos_iters, int vel_iters,
    const std::string& drive_type_str,
    bool local_poses)
{
    if(ptr) throw std::runtime_error("already finalized");
    if(!g_physics) throw std::runtime_error("call init_foundation() first");
    int n_links=(int)link_names.size(), n_joints=(int)joints.size();
    if(n_joints!=n_links-1) throw std::runtime_error("joint count mismatch");

    auto mb=mass_arr.unchecked<1>(), ib=inert_arr.unchecked<1>();
    auto pb=pos_arr.unchecked<1>(), qb=quat_arr.unchecked<1>();
    auto parb=parent_arr.unchecked<1>();
    auto cmpb=com_pos.unchecked<1>(), cmqb=com_quat.unchecked<1>();
    auto axb=axis_arr.unchecked<1>();
    auto lob=lower_arr.unchecked<1>(), upb=upper_arr.unchecked<1>();
    auto fb=fric_arr.unchecked<1>();

    // Validate axes: what add_joint stored must match what finalize received
    for(int i=0;i<n_joints;i++) {
        int got=axb(i);
        if((joint_axes[i]==PxArticulationAxis::eTWIST  && got!=0) ||
           (joint_axes[i]==PxArticulationAxis::eSWING1 && got!=1) ||
           (joint_axes[i]==PxArticulationAxis::eSWING2 && got!=2))
            throw std::runtime_error("joint_axis["+std::to_string(i)+"] mismatch");
    }

    // Create empty articulation
    ptr = g_physics->createArticulationReducedCoordinate();
    if(!ptr) throw std::runtime_error("createArticulationReducedCoordinate failed");

    // link pose passed to createLink(parent, pose): PhysX interprets `pose` as
    // PARENT-RELATIVE and propagates it parent→child.  Passing world-accumulated
    // poses double-applies the parent transforms at every level below the root,
    // displacing deep links (ankle at ~2-3 m) and their collision shapes.
    // local_poses=true passes the raw MJCF body poses (already parent-local).
    std::vector<PxTransform> world_poses(n_links);
    for(int i=0;i<n_links;i++) {
        PxTransform body(PxVec3(pb(i*3),pb(i*3+1),pb(i*3+2)),
                         PxQuat(qb(i*4+1),qb(i*4+2),qb(i*4+3),qb(i*4+0)));
        int p=parb(i);
        if(p>=0) world_poses[i] = world_poses[p].transform(body);
        else     world_poses[i] = body;
    }
    for(int i=0;i<n_links;i++) {
        int p=parb(i);
        PxArticulationLink *parent = (p>=0) ? links[p] : nullptr;
        PxTransform body(PxVec3(pb(i*3),pb(i*3+1),pb(i*3+2)),
                         PxQuat(qb(i*4+1),qb(i*4+2),qb(i*4+3),qb(i*4+0)));
        links[i] = ptr->createLink(parent, local_poses ? body : world_poses[i]);
        if(!links[i]) throw std::runtime_error("createLink failed: "+link_names[i]);

        links[i]->setMass(mb(i));
        links[i]->setMassSpaceInertiaTensor(PxVec3(ib(i*3),ib(i*3+1),ib(i*3+2)));
        links[i]->setCMassLocalPose(PxTransform(
            PxVec3(cmpb(i*3),cmpb(i*3+1),cmpb(i*3+2)),
            PxQuat(cmqb(i*4+1),cmqb(i*4+2),cmqb(i*4+3),cmqb(i*4+0))));

        // NOTE: setParentPose/setChildPose tested (T3, Route A variants)
        // — all cause NaN.  createLink(world_poses) joint frame is what
        // PhysX uses internally; we cannot change it safely.
    }

    // Configure joints (joint i connects the non-root link i+1 to its parent)
    for(int i=0;i<n_joints;i++) {
        int child=i+1;
        auto *joint = links[child]->getInboundJoint();
        joints[i] = joint;
        PxArticulationAxis::Enum ax = joint_axes[i];

        joint->setJointType(PxArticulationJointType::eREVOLUTE);
        joint->setMotion(ax, PxArticulationMotion::eLIMITED);
        // Lock the OTHER two axes — default is eFREE, causing instability
        if(ax!=PxArticulationAxis::eTWIST)  joint->setMotion(PxArticulationAxis::eTWIST,  PxArticulationMotion::eLOCKED);
        if(ax!=PxArticulationAxis::eSWING1) joint->setMotion(PxArticulationAxis::eSWING1, PxArticulationMotion::eLOCKED);
        if(ax!=PxArticulationAxis::eSWING2) joint->setMotion(PxArticulationAxis::eSWING2, PxArticulationMotion::eLOCKED);

        PxArticulationLimit limit; limit.low=lob(i); limit.high=upb(i);
        joint->setLimitParams(ax, limit);

        joint->setFrictionCoefficient(fb(i));

        PxArticulationDrive drive;
        drive.stiffness = drive_cfgs[i].kp;
        drive.damping   = drive_cfgs[i].kd;
        drive.maxForce  = drive_cfgs[i].force;
        drive.driveType = parse_drive_type(drive_type_str);
        joint->setDriveParams(ax, drive);
    }

    ptr->setSolverIterationCounts((PxU32)pos_iters, (PxU32)vel_iters);
}

// ── State queries (per-axis) ──
py::array_t<float> Articulation::get_joint_positions() {
    if(!ptr) throw std::runtime_error("not finalized");
    int n=(int)joints.size();
    py::array_t<float> r({n}); auto b=r.mutable_unchecked<1>();
    for(int i=0;i<n;i++) b(i)=joints[i]->getJointPosition(joint_axes[i]);
    return r;
}
py::array_t<float> Articulation::get_joint_velocities() {
    if(!ptr) throw std::runtime_error("not finalized");
    int n=(int)joints.size();
    py::array_t<float> r({n}); auto b=r.mutable_unchecked<1>();
    for(int i=0;i<n;i++) b(i)=joints[i]->getJointVelocity(joint_axes[i]);
    return r;
}
auto Articulation::get_root_world_pose() -> std::pair<py::array_t<float>,py::array_t<float>> {
    if(!ptr) throw std::runtime_error("not finalized");
    PxTransform t=ptr->getRootGlobalPose();
    return {v3_to_np(t.p), quat_to_np(t.q)};
}
auto Articulation::get_root_world_velocity() -> std::pair<py::array_t<float>,py::array_t<float>> {
    if(!ptr) throw std::runtime_error("not finalized");
    return {v3_to_np(ptr->getRootLinearVelocity()), v3_to_np(ptr->getRootAngularVelocity())};
}
auto Articulation::get_link_world_pose(int idx) -> std::pair<py::array_t<float>,py::array_t<float>> {
    if(!ptr) throw std::runtime_error("not finalized");
    if(idx<0||idx>=(int)links.size()) throw std::out_of_range("link index");
    // Use per-link getGlobalPose
    PxTransform t = links[idx]->getGlobalPose();
    return {v3_to_np(t.p), quat_to_np(t.q)};
}

// ── State mutation (per-axis) ──
void Articulation::set_root_world_pose(py::array_t<float> p, py::array_t<float> q) {
    if(!ptr) throw std::runtime_error("not finalized");
    ptr->setRootGlobalPose(np_to_xf(p,q), false);
    ptr->updateKinematic(PxArticulationKinematicFlag::ePOSITION);
}
void Articulation::set_root_world_velocity(py::array_t<float> lin, py::array_t<float> ang) {
    if(!ptr) throw std::runtime_error("not finalized");
    ptr->setRootLinearVelocity(np_to_v3(lin), true);
    ptr->setRootAngularVelocity(np_to_v3(ang), true);
}
void Articulation::set_joint_positions(py::array_t<float> qpos) {
    if(!ptr) throw std::runtime_error("not finalized");
    auto b=qpos.unchecked<1>(); int n=(int)joints.size();
    for(int i=0;i<n;i++) joints[i]->setJointPosition(joint_axes[i], b(i));
    ptr->updateKinematic(PxArticulationKinematicFlag::ePOSITION);
}
void Articulation::set_joint_velocities(py::array_t<float> qvel) {
    if(!ptr) throw std::runtime_error("not finalized");
    auto b=qvel.unchecked<1>(); int n=(int)joints.size();
    for(int i=0;i<n;i++) joints[i]->setJointVelocity(joint_axes[i], b(i));
    ptr->updateKinematic(PxArticulationKinematicFlag::eVELOCITY);
}
void Articulation::set_joint_drive_targets(py::array_t<float> tgts) {
    if(!ptr) throw std::runtime_error("not finalized");
    auto b=tgts.unchecked<1>(); int n=(int)joints.size();
    for(int i=0;i<n;i++) joints[i]->setDriveTarget(joint_axes[i], b(i), true);
}
void Articulation::set_joint_drive_params(int idx, float kp, float kd, float fl,
                                          const std::string& drive_type_str) {
    if(!ptr) throw std::runtime_error("not finalized");
    if(idx<0||idx>=(int)joints.size()) throw std::out_of_range("joint index");
    PxArticulationDrive d; d.stiffness=kp; d.damping=kd; d.maxForce=fl;
    d.driveType=parse_drive_type(drive_type_str);
    joints[idx]->setDriveParams(joint_axes[idx], d);
}
void Articulation::set_solver_iterations(int pi, int vi) {
    if(!ptr) throw std::runtime_error("not finalized");
    ptr->setSolverIterationCounts((PxU32)pi, (PxU32)vi);
}
void Articulation::set_joint_armatures(py::array_t<float> armatures) {
    if(!ptr) throw std::runtime_error("not finalized");
    auto b=armatures.unchecked<1>(); int n=(int)joints.size();
    if(b.shape(0)!=n) throw std::runtime_error("armatures length mismatch");
    for(int i=0;i<n;i++) joints[i]->setArmature(joint_axes[i], b(i));
}
void Articulation::set_joint_velocity_limits(py::array_t<float> vlims) {
    if(!ptr) throw std::runtime_error("not finalized");
    auto b=vlims.unchecked<1>(); int n=(int)joints.size();
    if(b.shape(0)!=n) throw std::runtime_error("velocity limits length mismatch");
    for(int i=0;i<n;i++) joints[i]->setMaxJointVelocity(joint_axes[i], b(i));
}
void Articulation::set_max_depenetration_velocity(float v) {
    if(!ptr) throw std::runtime_error("not finalized");
    for(auto *l : links) l->setMaxDepenetrationVelocity(v);
}
void Articulation::set_joint_parent_pose(int joint_idx, py::array_t<float> pos, py::array_t<float> quat) {
    if(!ptr) throw std::runtime_error("not finalized");
    if(joint_idx<0||joint_idx>=(int)joints.size()) throw std::out_of_range("joint index");
    PxTransform pp(np_to_v3(pos), np_to_quat(quat));
    joints[joint_idx]->setParentPose(pp);
}
void Articulation::set_joint_child_pose(int joint_idx, py::array_t<float> pos, py::array_t<float> quat) {
    if(!ptr) throw std::runtime_error("not finalized");
    if(joint_idx<0||joint_idx>=(int)joints.size()) throw std::out_of_range("joint index");
    PxTransform cp(np_to_v3(pos), np_to_quat(quat));
    joints[joint_idx]->setChildPose(cp);
}

void Articulation::add_link_torque(int link_idx, float x, float y, float z) {
    if(!ptr) throw std::runtime_error("not finalized");
    if(link_idx<0||link_idx>=(int)links.size()) throw std::out_of_range("link index");
    links[link_idx]->addTorque(PxVec3(x, y, z));
}

// ── Shapes ──
void Articulation::attach_sphere(int lidx, float r, py::array_t<float> p, py::array_t<float> q){
    PxSphereGeometry g(r); auto *m=get_mat();
    PxShape *s=g_physics->createShape(g,*m,true);
    s->setLocalPose(np_to_xf(p,q)); s->setContactOffset(0.02f);
    links[lidx]->attachShape(*s);
}
void Articulation::attach_box(int lidx, float hx,float hy,float hz, py::array_t<float> p, py::array_t<float> q){
    PxBoxGeometry g(hx,hy,hz); auto *m=get_mat();
    PxShape *s=g_physics->createShape(g,*m,true);
    s->setLocalPose(np_to_xf(p,q));
    // contactOffset must be < the smallest half-extent (foot box hz=0.015).
    // 0.02 > 0.015 inflates the PCM manifold past the thin dimension and the
    // positional correction ejects the robot at ~10 m/s on first contact.
    s->setContactOffset(hz<0.02f ? 0.005f : 0.02f);
    links[lidx]->attachShape(*s);
}
void Articulation::attach_capsule(int lidx, float r, float hh, py::array_t<float> p, py::array_t<float> q){
    PxCapsuleGeometry g(r,hh); auto *m=get_mat();
    PxShape *s=g_physics->createShape(g,*m,true);
    s->setLocalPose(np_to_xf(p,q)); s->setContactOffset(0.02f);
    links[lidx]->attachShape(*s);
}

// ═══════════════════════════════════════════════════════════════════════
// Module
// ═══════════════════════════════════════════════════════════════════════
PYBIND11_MODULE(physx_core, m) {
    m.doc()="PhysX 5 SDK — pybind11 wrapper for SONIC G1 training";

    py::class_<Scene>(m,"Scene")
        .def("simulate",&Scene::simulate,py::arg("dt"))
        .def("fetch_results",&Scene::fetch_results,py::arg("block")=true)
        .def("create_material",&Scene::create_material,
             py::arg("static_friction"),py::arg("dynamic_friction"),py::arg("restitution"))
        .def("add_articulation",&Scene::add_articulation,py::arg("art"))
        .def("add_ground_plane",&Scene::add_ground_plane,
             py::arg("material_idx"),py::arg("normal"))
        .def("get_contacts",&Scene::get_contacts)
        .def("clear_contacts",&Scene::clear_contacts);

    py::class_<Articulation>(m,"Articulation")
        .def(py::init<>())
        .def("add_link",&Articulation::add_link,
             py::arg("parent_idx"),py::arg("name"))
        .def("add_joint",&Articulation::add_joint,
             py::arg("parent_idx"),py::arg("child_idx"),py::arg("axis"),
             py::arg("kp")=100.0f,py::arg("kd")=5.0f,py::arg("force_limit")=50.0f)
        .def("finalize",&Articulation::finalize,
             py::arg("link_masses"),py::arg("link_diag_inertia"),
             py::arg("link_local_pos"),py::arg("link_local_quat"),
             py::arg("link_parents"),
             py::arg("link_com_pos"),py::arg("link_com_quat"),
             py::arg("joint_axis"),
             py::arg("joint_lower"),py::arg("joint_upper"),
             py::arg("joint_friction"),
             py::arg("position_iters")=8,py::arg("velocity_iters")=1,
             py::arg("drive_type")="ACCELERATION",
             py::arg("local_poses")=false)
        .def("get_joint_positions",&Articulation::get_joint_positions)
        .def("get_joint_velocities",&Articulation::get_joint_velocities)
        .def("get_root_world_pose",&Articulation::get_root_world_pose)
        .def("get_root_world_velocity",&Articulation::get_root_world_velocity)
        .def("get_link_world_pose",&Articulation::get_link_world_pose,py::arg("idx"))
        .def("get_link_index",&Articulation::get_link_index,py::arg("name"))
        .def("get_link_actor_ptr",&Articulation::get_link_actor_ptr,py::arg("idx"))
        .def_property_readonly("num_joints",&Articulation::num_joints)
        .def_property_readonly("num_links",&Articulation::num_links)
        .def("set_root_world_pose",&Articulation::set_root_world_pose,
             py::arg("pos"),py::arg("quat"))
        .def("set_root_world_velocity",&Articulation::set_root_world_velocity,
             py::arg("lin"),py::arg("ang"))
        .def("set_joint_positions",&Articulation::set_joint_positions,py::arg("qpos"))
        .def("set_joint_velocities",&Articulation::set_joint_velocities,py::arg("qvel"))
        .def("set_joint_drive_targets",&Articulation::set_joint_drive_targets,py::arg("targets"))
        .def("set_joint_drive_params",&Articulation::set_joint_drive_params,
             py::arg("idx"),py::arg("kp"),py::arg("kd"),py::arg("force_limit"),
             py::arg("drive_type")="ACCELERATION")
        .def("set_solver_iterations",&Articulation::set_solver_iterations,
             py::arg("pos_iters"),py::arg("vel_iters"))
        .def("set_joint_armatures",&Articulation::set_joint_armatures,
             py::arg("armatures"))
        .def("set_joint_velocity_limits",&Articulation::set_joint_velocity_limits,
             py::arg("vlims"))
        .def("set_max_depenetration_velocity",&Articulation::set_max_depenetration_velocity,
             py::arg("v")=1.0f)
        .def("set_joint_parent_pose",&Articulation::set_joint_parent_pose,
             py::arg("joint_idx"),py::arg("pos"),py::arg("quat"))
        .def("set_joint_child_pose",&Articulation::set_joint_child_pose,
             py::arg("joint_idx"),py::arg("pos"),py::arg("quat"))
        .def("add_link_torque",&Articulation::add_link_torque,
             py::arg("link_idx"),py::arg("x"),py::arg("y"),py::arg("z"))
        .def("attach_sphere",&Articulation::attach_sphere,
             py::arg("link_idx"),py::arg("radius"),py::arg("pos"),py::arg("quat"))
        .def("attach_box",&Articulation::attach_box,
             py::arg("link_idx"),py::arg("hx"),py::arg("hy"),py::arg("hz"),
             py::arg("pos"),py::arg("quat"))
        .def("attach_capsule",&Articulation::attach_capsule,
             py::arg("link_idx"),py::arg("radius"),py::arg("half_height"),
             py::arg("pos"),py::arg("quat"));

    m.def("init_foundation",[](){
        if(g_foundation) return;
        g_allocator=new PxDefaultAllocator(); g_error_cb=new MinimalErrorCallback();
        g_foundation=PxCreateFoundation(PX_PHYSICS_VERSION,*g_allocator,*g_error_cb);
        if(!g_foundation) throw std::runtime_error("PxCreateFoundation failed");
        g_physics=PxCreatePhysics(PX_PHYSICS_VERSION,*g_foundation,PxTolerancesScale(),false,nullptr);
        if(!g_physics) throw std::runtime_error("PxCreatePhysics failed");
    });
    m.def("release_foundation",[](){
        for(auto *s:g_scenes){if(s)s->release();} g_scenes.clear();
        if(g_physics){g_physics->release();g_physics=nullptr;}
        if(g_foundation){g_foundation->release();g_foundation=nullptr;}
        delete g_error_cb;g_error_cb=nullptr;delete g_allocator;g_allocator=nullptr;
    });
    m.def("create_scene",[](py::array_t<float> g,const std::string &st,int pi,int vi)->Scene*{
        if(!g_physics) throw std::runtime_error("call init_foundation() first");
        PxSceneDesc sd(g_physics->getTolerancesScale()); sd.gravity=np_to_v3(g);
        sd.cpuDispatcher=PxDefaultCpuDispatcherCreate(1); sd.filterShader=filter_shader;
        sd.simulationEventCallback=&g_debug_cb;
        sd.bounceThresholdVelocity=2.0f; sd.frictionType=PxFrictionType::ePATCH;
        sd.solverType=(st=="TGS")?PxSolverType::eTGS:PxSolverType::ePGS;
        PxScene *pxs=g_physics->createScene(sd);
        if(!pxs) throw std::runtime_error("createScene failed");
        g_scenes.push_back(pxs);
        return new Scene{pxs,pi,vi};
    },py::arg("gravity"),py::arg("solver_type")="TGS",
       py::arg("position_iters")=8,py::arg("velocity_iters")=1,
       py::return_value_policy::take_ownership);

    m.def("_test_quat_roundtrip",[](py::array_t<float> q){return quat_to_np(np_to_quat(q));});
}
