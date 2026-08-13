# PhysX Training Plan v8 — 4096-Env Full-Scale Training

**Date:** 2026-08-08
**Status:** Ready to launch
**Goal:** Train SONIC 37M policy on PhysX 5 CPU (eACCELERATION) to match/exceed Isaac Sim training quality.

---

## 1. Background: Why v1-v7 Failed

| | MuJoCo (successful) | PhysX v1-v7 |
|---|---|---|
| envs | 4096 | 64 |
| DDP processes | 16 | 1 (v1-v5) / 16 (v7) |
| steps/iter | 1.57M | 1,536 (v1-v5) / 24,576 (v7) |
| throughput | 1024× | baseline ≡ 1× |

BC v6 proved the encoder CAN adapt to PhysX observations (g1_recon 0.64→0.0001). The decoder never got enough data for PPO to reshape action distribution — v7 entropy stuck at 15.44 (near-random), rewards 0.0, length NaN.

**PhysXEnvManager already supports 4096 envs.** Code-zero change from v7: just `num_envs=4096`.

---

## 2. Isaac Sim Configuration Alignment

### Reward Function (13 items)

| # | Item | Isaac weight | PhysX current | Match? |
|---|------|-------------|--------------|--------|
| r1 | anchor_pos | 0.5, σ=0.3 | 0.5, σ²=0.09 | ✅ |
| r2 | anchor_ori | 0.5, σ=0.4 | 0.5, σ²=0.16 | ✅ |
| r3 | body_pos | 1.0, σ=0.3 | 1.0 | ✅ |
| r4 | body_ori | 1.0, σ=0.4 | 1.0 | ✅ |
| r5 | body_linvel | 1.0, σ=1.0 | 1.0 | ✅ |
| r6 | body_angvel | 1.0, σ=3.14 | 1.0, /9.86 | ✅ |
| r7 | action_rate_l2 | -0.1 | -0.1 | ✅ |
| r8 | joint_limit | -10.0 | -10.0 | ✅ |
| r9 | undesired_contacts | -0.1 | 0.0 (TODO) | ❌ |
| r10 | anti_shake | -0.005 | -0.005 | ✅ |
| r11 | vr_5point | 2.0, σ=0.1 | 2.0 | ✅ |
| r12 | feet_acc | -2.5e-6 (Cartesian) | -2.5e-9 (joint) | ❌ 1000× |
| r13 | alive_bonus | 0 | 4.0 | ❌ |

### Termination Thresholds

| Threshold | Isaac Sim | PhysX current | Gap |
|-----------|-----------|--------------|-----|
| anchor_ori | 0.2 | 0.35 | 1.75× |
| ankle_pos | 0.2 | 0.35 | 1.75× |
| ankle_h_mult | n/a | 1.5 | extra |
| anchor_pos | 0.15 | missing | - |

### GAPS to Fix (Phase 0)

1. **r12**: `-2.5e-9` → `-2.5e-6` (1 line)
2. **r9**: returns 0.0 — keep as-is until PhysX contact API, negligible impact
3. **Term thresholds**: 0.35/0.35/1.5 → 0.2/0.2/1.0 (3 lines)
4. **alive_bonus**: 4.0 → 0 (config param)

**After Phase 0 fixes: PhysX config = Isaac Sim config (functionally equivalent)**

---

## 3. Training Phases

### Phase 0: Environment Fixes
- Fix r12 coefficient
- Tighten termination thresholds to Isaac Sim values
- Set alive_bonus=0 in default config
- Git commit

### Phase 1: Encoder Adaptation (BC Warmup)
```
num_envs: 4096
DDP: 16 (accelerate launch)
ppo_loss_coef: 0
g1_recon: 1.0
skip_termination: True
alive_bonus: 0
ignore_terminations: True
num_learning_iterations: 500
```
- Start from SONIC release checkpoint
- Goal: g1_recon < 0.001
- ~300 iter expected (~8 min at 1.5s/iter)

### Phase 2: PPO Training (Strict Isaac Thresholds)
```
num_envs: 4096
DDP: 16
ppo_loss_coef: 1.0
skip_termination: False
alive_bonus: 0
ignore_terminations: False
num_learning_iterations: 3000
```
- Start from Phase 1 checkpoint
- Strict Isaac thresholds: ORI=0.2, ANK=0.2
- Goal: episode length > 20, reward positive, entropy decreasing

### Phase 2b (Emergency): Death Spiral Rescue
If PPO with strict thresholds fails:
- alive_bonus=1.0 for 100 iter → back to 0
- OR threshold relaxation: 0.2→0.3→0.2 over 500 iter

### Phase 3: Production Training
```
g1_recon: 0.01 (light encoder continuation)
```
- Goal: 3000+ iter, Isaac Sim deterministic eval > 80 steps
- Final checkpoint directly usable in Isaac Sim

---

## 4. Expected Throughput

| Phase | envs × DDP | steps/iter | time/iter | total time (500 iter) |
|-------|-----------|------------|-----------|----------------------|
| Phase 1/2 | 4096 × 16 | 1.57M | ~1.5s | ~12 min |

Compares to MuJoCo's successful 4096×16 = 1.57M steps/iter.

---

## 5. Monitoring & Evaluation

### TensorBoard
- Port forwarding for remote access
- Key metrics: loss/policy_avg, loss/entropy_avg, objective/rewards, objective/length

### Periodic Evaluation
- Every 100 iterations: run deterministic inference on 1 env
- Render trajectory frames to video
- Check: episode length, reward, visual quality

### Disk Monitoring
- Check disk space every 50 iterations
- Auto-clean old checkpoints (keep last 5)
- Alert if usage > 80%

### Training Health
- Continuous monitoring for: NaN, worker crash, memory OOM, barrier timeout
- Auto-restart with exponential backoff on crash
- Push notification on critical alerts

---

## 6. Launch Command

```bash
cd /root/GR00T-WholeBodyControl
export SONIC_PHYSX_ENV=1

accelerate launch --num_processes=16 --num_machines=1 \
    gear_sonic/train_agent_trl.py \
    +exp=manager/universal_token/all_modes/sonic_release \
    exp_base=physx_ppo exp_var=v8 \
    project_name=TRL_G1_PhysX \
    checkpoint=/root/sonic_release/last.pt \
    num_envs=4096 \
    algo.config.num_learning_iterations=500 \
    headless=true use_wandb=false
```

---

## 7. Risk Mitigation

| Risk | Probability | Mitigation |
|------|------------|------------|
| 4096 envs OOM (>512GB) | Low | Estimated 122GB, >300GB margin |
| Worker physx_core fork conflict | Low | Verified fork+import pattern (physx-multi-process-solved) |
| PPO death spiral at strict thresholds | Medium | Phase 2b: temporary alive_bonus |
| Disk full during training | Medium | Periodic cleanup, monitoring |
| 16-process Barrier desync | Low | Barrier per-worker, crash recovery built-in |
