# PhysX Training Log

**Last updated:** 2026-08-11 (afternoon session)
**Current status:** BC warmup from Phase B ckpt RUNNING (BC coef=1.0, action_trust=0, 4096 envs, iter ~5/100)
**Next action:** BC 跑完 100 iter → 逐步提升 action_trust (0→0.5→1.0) → 切 PPO

---

## Current State Summary

| Training | Phase | Status | Key Result |
|----------|-------|--------|------------|
| BC v4 | BC warmup | Complete | 0 iter (empty log) |
| BC v5 | PPO | Complete | iter 680, clean |
| BC v7 | BC/PPO | Dead | iter 1038, 100% NaN, killed |
| Phase B PPO | PPO finetune | Never started | Empty log |
| v8 P1 | Encoder adaptation | **Passed** | g1_recon=0.0006 @ iter 198 |
| v8 P2 | PPO strict thresholds | **Failed** | length=1.0 (dies step 1) |
| v8 P3 | skip_termination warmup | **Failed** | rewards=0 (no signal) |
| v8 P4 | Strict re-evaluation | **Failed** | length=1.0 (dies step 1) |
| v9 P1 | Encoder adaptation | **Passed** | g1_recon=0.0006 @ iter 200 |
| v9 P2 | Loose PPO | **Failed** | length=1.0, 150 iter, no improvement |
| v10 BC | BC+PPO joint (skip_term) | **Failed** | value=3.4M, advantage=0, lr collapsed |
| v11 | ignore_terminations PPO | **Partial** | value=129 stable, but length stuck at 1.0 |
| v12 | ignore_term + BC loss | **Failed** | BC loss 0.09 stuck, encoder not adapted |
| v13 S1 | Encoder adaptation | **Passed** | g1_recon=0.0006 @ 200 iter |
| v13 S2 | BC+PPO from adapted encoder | **Partial** | BC loss 0.12 stuck, length=1.0 |
| v14 | g1_dyn decoder reinit | **Failed** | Architecture error — decoder too small for PhysX dynamics |
| v14 s2a | BC warmup (action_trust=0) | **Passed** | policy_avg=0.0007, length=5.1 @ 300 iter |
| v15 BC | BC warmup regression test | **Passed** | policy_avg=0.00067, length=7.84 @ 300 iter |
| **v1 prod** | **4096-env full PPO** | **Completed** | iter 1248, rewards 35.4, length 26.6 (BC baseline ×3.4), no NaN |
| **v2 tight** | **Threshold tighten 0.30** | **RUNNING** | From v1 iter 1000 ckpt, ori=0.30 ank_pos=0.30, adapting to stricter bounds |
| **v3 root-vel** | **Root velocity + NaN guard fix ⭐** | **Completed** | **Identified 2 root causes: reset() missed root vel, NaN leaked through IEEE 754** |
| **v3 verify** | **Physics validation post-fix** | **Completed** | **10 iter, length 9.2-9.5 stable, 0 NaN (was 2.59→1.0 pre-fix)** |
| **v4 BC** | **BC warmup from Phase B ckpt** | **RUNNING** | **BC coef=1.0, action_trust=0, ~5/100 iter, length stable ~9.2** |

---

## Run History

### 2026-08-08 — v8 Full Training Campaign

**Config:** 4096 envs, 1024 workers, 16 DDP, PhysX 5 Direct API eACCELERATION
**Model:** SONIC 37M UniversalTokenModule
**Checkpoint base:** SONIC release `last.pt`

#### P1: Encoder Adaptation (v8, exp_var=v8)

| Item | Value |
|------|-------|
| Launch | 2026-08-08 ~10:00 UTC |
| Duration | 198 iterations |
| SONIC_PHYSX_ENC_ADAPT | 1 |
| ppo_loss_coef | 0 |
| g1_recon weight | 1.0 |
| skip_termination | True |

**Final metrics:**
- g1_recon: 0.0006 @ iter 198
- policy_avg: 0.0030 @ iter 198
- Iteration time: ~11s

**Verdict: PASSED.** Encoder successfully adapted to PhysX observations.

#### P2: PPO Strict Isaac Thresholds (v8_p2)

| Item | Value |
|------|-------|
| Launch | 2026-08-08 ~15:51 UTC |
| Duration | 200 iterations |
| Checkpoint | v8 last.pt |
| Encoder adaptation | OFF |
| skip_termination | OFF |
| Thresholds | ori=0.2, ank_pos=0.2, ank_h=1.0 (Isaac strict) |
| alive_bonus | 0 |

**Final metrics:**
- length: 1.0000 (all 4096 envs die in step 1)
- rewards: ~-1.4
- policy_avg: ~-0.005

**Verdict: FAILED.** Policy cannot survive past step 1 under strict Isaac thresholds.
GAE bootstrap broken — value function never sees step 2+.

#### P3: skip_termination Warmup (v8_p3)

| Item | Value |
|------|-------|
| Launch | 2026-08-08 ~17:17 UTC |
| Duration | 200 iterations |
| Checkpoint | v8_p2 last.pt |
| SONIC_PHYSX_SKIP_TERM | 1 |
| Encoder adaptation | OFF |

**Final metrics:**
- rewards: 0.0 throughout
- No learning signal

**Verdict: FAILED.** skip_termination prevents env reset but gives zero rewards.
Policy never receives any learning signal.

#### P4: Strict Re-evaluation (v8_p4)

| Item | Value |
|------|-------|
| Launch | 2026-08-08 ~17:52 UTC |
| Duration | 200 iterations |
| Checkpoint | v8_p3 last.pt |
| skip_termination | OFF |
| Thresholds | ori=0.2, ank_pos=0.2, ank_h=1.0 (Isaac strict) |

**Final metrics (iter 155 TB):**
- length: 1.0000
- rewards: -1.3783
- policy_avg: -0.0048
- g1_recon: 0.0541

**Verdict: FAILED.** Same death-in-step-1 outcome as P2.

---

### Earlier Runs (BC v4 – v7)

| Run | Date | Config | Result |
|-----|------|--------|--------|
| BC v4 | 2026-08-08 | BC warmup | 0 iter, empty log |
| BC v5 | 2026-08-08 | PPO training | iter 680, clean, complete |
| BC v7 | 2026-08-08 | BC/PPO | No TB — never ran |
| Phase B PPO | 2026-08-08 | PPO finetune | Log empty — never started |

---

### 2026-08-09 — v9 Gradual Threshold Tightening

**Config:** 4096 envs, 1024 workers, 16 DDP, PhysX 5 Direct API eACCELERATION
**Model:** SONIC 37M UniversalTokenModule
**Strategy:** 5-phase threshold schedule: skip_termination → 0.6 → 0.4 → 0.3 → 0.2 (Isaac)
**Code changes:** `physx_env.py` + `train_agent_trl.py` — thresholds configurable via `SONIC_PHYSX_ORI_THRESH`, `SONIC_PHYSX_ANK_POS_THRESH`, `SONIC_PHYSX_ANK_H_MULT` env vars

#### v9 P1: Encoder Adaptation (PASSED)

| Item | Value |
|------|-------|
| Launch | 2026-08-09 01:36 UTC |
| Log | physx_v9_p1_20260809_013651.log |
| Duration | 200 iterations |
| Checkpoint | SONIC release last.pt |
| Iteration time | ~8s |
| g1_recon | **0.0006 @ iter 200** |
| policy_avg | 0.0041 @ iter 200 |
| Errors | 0 |

**Verdict: PASSED.** g1_recon=0.0006 < 0.001 target. Encoder successfully adapted.
Checkpoint: `physx_ppo_v9_p1-20260809_013706/last.pt`

#### v9 P2: Loose PPO (FAILED — length=1.0)

| Item | Value |
|------|-------|
| Launch | 2026-08-09 02:08 UTC |
| Log | physx_v9_p2_20260809_020828.log |
| Checkpoint | v9 P1 last.pt |
| Thresholds | ori=0.6, ank_pos=0.6, ank_h=3.0 |
| Target iter | 150 |

**Final metrics:**
- length: 1.00000 (all 150 iters, 9.8M episodes — zero improvement)
- rewards: -1.20183
- entropy: 13.79
- errors: 0
- checkpoint: `physx_ppo_v9_p2-20260809_020845/last.pt`

**Verdict: FAILED.** Even 3×-loose thresholds can't prevent step-1 death. Root cause: P1 encoder adaptation trained with `ppo_loss_coef=0` + `skip_termination=True`, so the policy produces effectively random actions. It never learned to stand, let alone walk.

#### v9 P3-P5: CANCELLED

P3-P5 are pointless when P2 fails at the loosest threshold. The gradual tightening strategy requires the policy to survive P2 first.

---

### 2026-08-09 — v9 Post-Mortem

The chicken-and-egg is deeper than initially thought:
1. P1 (encoder adaptation) → policy doesn't learn locomotion actions (ppo_loss_coef=0)
2. P2 (loose PPO from P1 ckpt) → random actions → step-1 death even at 3× thresholds
3. The gap between "encoder adapted" and "can stand" is too large for PPO to cross from scratch

**Decision: Option A — BC+PPO Joint Training (v10)**

---

### 2026-08-09 — v10 BC+PPO Joint Training

**Strategy:** Start from SONIC release checkpoint, enable `skip_termination` so the policy gets full 24-step trajectories, and jointly train with PPO (tracking rewards) + g1_recon (encoder adaptation). The key insight: with skip_termination, PPO rewards ARE computed (r1-r13 in `physx_env.py:358-394`) — the policy gets meaningful gradient signal from reference motion tracking. After the policy stabilizes on PhysX, remove skip_termination and apply gradual threshold tightening.

**Why this should work where v9 failed:**
- v9 P1: ppo_loss_coef=0 → decoder frozen, random actions
- v10 BC: ppo_loss_coef=1.0 → decoder learns from tracking rewards
- skip_termination → full 24-step trajectories → GAE works properly
- SONIC release ckpt → encoder starts with reasonable representations

#### v10 BC: Joint Training (FAILED — value explosion, zero gradients)

| Item | Value |
|------|-------|
| Launch | 2026-08-09 11:43 UTC |
| Killed | 2026-08-09 ~12:00 UTC (~117 iter) |
| Log | physx_v10_bc_20260809_114352.log |
| Checkpoint | SONIC release last.pt |
| SONIC_PHYSX_SKIP_TERM | 1 |
| ppo_loss_coef | 1.0 (default, PPO active) |
| g1_recon coef | 0.01 (default) |

**Final TB metrics @ iter 95 (last TB write):**
- `objective/rewards`: 0.0000 — tracking rewards all zero
- `val/advantage_mean`: 0.0000 — zero advantage, PPO has no gradient
- `loss/value_avg`: 3,477,985 — value function exploded
- `loss/aux_g1_recon_avg`: 0.1072 — encoder polluted (v9 P1 was 0.0006)
- `lr`: 0.0000 — adaptive LR collapsed
- `objective/length`: nan — skip_termination means episodes never end

**Verdict: FAILED.** Root cause: `skip_termination=True` makes episodes infinite → no episode boundary → GAE bootstrap broken → value explosion. Additionally, tracking rewards r1-r6 are exp(-large_error²/σ) ≈ 0 when robot falls, and punishment terms (r7=-0.03/step, r12=-2.5e-6) are too weak to provide signal. The combined result: zero gradient, zero learning.

**Key lesson vs MuJoCo:** `skip_termination` is different from `ignore_terminations`. skip_term prevents the episode from ever ending (length=nan, GAE broken). ignore_terminations triggers termination-driven reset but ignores done in the trainer — episodes are finite, GAE bootstraps normally.

#### v10 P2-P5: CANCELLED

---

### 2026-08-09 — v11: ignore_terminations + BC Loss (MuJoCo E1 Strategy)

**Lessons from MuJoCo 13-round experiments applied to PhysX:**

1. **`skip_termination` → `ignore_terminations`**: Termination triggers normally → env reset → finite episodes → GAE bootstraps. Trainer ignores done, so policy survives rollout-boundary resets.
2. **BC loss on reference motion**: MSE(policy_action, reference_qpos) as auxiliary loss. g1_dyn decoder is directly supervised by reference joint angles — solves the "zero tracking reward" problem by providing gradient signal even when robot falls.
3. **Freeze encoder during BC phase**: Follow MuJoCo Phase E1 design — freeze encoder+g1_kin, only train g1_dyn decoder.
4. **Reward verification**: All tracking rewards (r1-r12) are computed correctly in `_compute_reward()` — verify with non-falling test.

**Implementation plan:**

| Change | File | Description |
|--------|------|-------------|
| A | `physx_env.py` | `skip_termination` → `ignore_terminations` semantics |
| B | `train_agent_trl.py` | `SONIC_PHYSX_BC_MODE` env var: freeze encoder, MSE loss on decoder |
| C | `ppo_trainer.py` or config | BC loss term: `policy_action` vs `reference_qpos` |

**Expected path:**
```
v11 BC (ignore_term + BC loss, 300 iter)
  → decoder learns to output ref-like actions
  → tracking rewards become non-zero (robot stays near ref)
  → PPO has meaningful gradient → entropy decreases
  → Gradual threshold tightening (P2-P5)
```

---

## Root Cause Analysis

**The Chicken-and-Egg Problem:**

1. Strict Isaac thresholds (ori=0.2, ank_pos=0.2) kill all policies in step 1
2. With length=1.0, GAE has no future steps to bootstrap from
3. Value function never sees step 2+, so it cannot learn to predict future returns
4. Without useful advantage estimates, PPO cannot improve the policy
5. The policy stays at its initial (random) quality and keeps dying in step 1

skip_termination was attempted as a workaround but failed because:
- rewards=0 means no learning signal
- The policy improves nothing because there's nothing to optimize

**This is a fundamental exploration problem:** the policy needs to survive to learn, but needs to learn to survive.

---

## Proposed Solution: Gradual Threshold Tightening (v9)

Instead of jumping directly to strict Isaac thresholds, gradually tighten them:

| Phase | Iterations | ori threshold | ank_pos threshold | ank_h_mult |
|-------|-----------|---------------|-------------------|------------|
| v9 P1 (BC warmup) | 0-200 | ∞ (skip_term) | ∞ (skip_term) | ∞ |
| v9 P2 (loose) | 200-350 | 0.6 | 0.6 | 3.0 |
| v9 P3 (medium) | 350-500 | 0.4 | 0.4 | 2.0 |
| v9 P4 (tight) | 500-650 | 0.3 | 0.3 | 1.5 |
| v9 P5 (Isaac) | 650-1000 | 0.2 | 0.2 | 1.0 |

**Approach:** At each threshold transition, load the previous checkpoint and continue training with the tighter thresholds. The policy learns to stay within each threshold band before the next tightening.

**Implementation needed:**
- [ ] `physx_env.py`: make thresholds configurable per-phase (env vars or config)
- [ ] `launch_v9_p1.sh`: BC warmup with skip_termination (encoder adaptation)
- [ ] `launch_v9_p2.sh`: PPO with loose thresholds
- [ ] `launch_v9_p3.sh`: PPO with medium thresholds
- [ ] `launch_v9_p4.sh`: PPO with tight thresholds
- [ ] `launch_v9_p5.sh`: PPO with Isaac strict thresholds

---

---

## 2026-08-10 — v11: ignore_terminations PPO

**Config:** `SONIC_PHYSX_IGNORE_TERM=1`, thresholds ori=0.6 ank=0.6 h=3.0, 200 iter from SONIC release ckpt
**Goal:** Prove skip→ignore fix works (finite episodes → proper GAE)

**Results (iter 95):**
| Metric | Value | Note |
|--------|-------|------|
| rewards | -3.33 → -1.60 | Slowly improving |
| value_avg | 129 | Stable (v10 was 3.4M) |
| GAE advantage | Non-zero | Bootstrap works |
| entropy | 12.8 → 13.5 | Rising |
| g1_recon | 0.016 | Decent |
| length | 1.0 | Still dying first step |

**Verdict: PARTIAL SUCCESS.** `ignore_terminations` fix confirmed — GAE healthy, value normal. But PPO alone insufficient — decoder not adapting to PhysX observation distribution. Entropy rising suggests policy exploring randomly rather than converging.

**Key finding:** Encoder is NOT adapted (g1_recon=0.016 from SONIC release baseline, not from dedicated adaptation). The v8 P1 encoder adaptation (g1_recon→0.0006) was NOT carried forward — this checkpoint was from a fresh SONIC release load.

---

## 2026-08-10 — v12: ignore_terminations + BC Loss

**Config:** `SONIC_PHYSX_BC_COEF=1.0` (new), same loose thresholds as v11
**Goal:** Add BC loss (MSE action_mean vs reference qpos) to provide direct behavioral supervision

**Code changes:**
- `physx_env.py`: `ref_action` in `_obs()` dict
- `physx_env_manager.py`: OBS_DIM 4336→4365, SHM concat includes ref_action
- `ppo_trainer_aux_loss.py`: `compute_bc_loss` + `bc_loss_coef` in `_compute_loss`
- `train_agent_trl.py`: `SONIC_PHYSX_BC_COEF` env var → `ref_action` in obs_dims

**Initial crash:** `KeyError: 'ref_action'` — physx_env.py not deployed (only env_manager was). Fixed by SCP physx_env.py.

**Results (iter 68):**
| Metric | Value | Trend (30-step) |
|--------|-------|----------------|
| BC loss | 0.09 | **Stuck** (0.089→0.093, no decrease) |
| g1_recon | 0.20→0.12 | ↓ Slowly improving |
| policy_avg | -0.008 | Near zero (BC dominates) |
| value_avg | 51→21 | ↓ Normal convergence |
| entropy | 13.26 | → Stable |
| rewards | -2.5→-1.6 | ↑ Weak improvement |
| **length** | **1.0** | **68 iterations, never broke 1.0** |

**Verdict: FAILED.** BC loss stuck at 0.09 (RMSE≈0.3rad≈17°/joint). Encoder never adapted (g1_recon=0.12 vs v8 P1's 0.0006 — 200× worse). Without correct latent tokens, decoder cannot map BC supervision onto correct physical context.

---

## 2026-08-10 — Key Findings

(This section collects insights discovered across multiple experiments.)

### h_thresh Root Cause of length=1.0

v14 BC training got `length=1.0` even with loose thresholds. Root cause: **hardcoded `h_thresh=0.15m`** in `_check_termination()`. Every ~9 steps the robot's torso height dropped below 0.15m (due to PhysX integration drift), triggering immediate termination.

**Fix:** Make `h_thresh` configurable via `_h_thresh` attribute + `SONIC_PHYSX_H_THRESH` env var. BC warmup sets `h_thresh=10.0` (effectively disabled), PPO training gates properly.

### ignore_terminations vs skip_termination Semantics

| | skip_termination | ignore_terminations |
|---|---|---|
| Episode ends | Never | Yes (env resets internally) |
| GAE bootstrap | Broken | Normal |
| Trainer sees done | No | No (ignored) |
| Use case | Encoder adaptation only | PPO/BC training |
| Safe for | g1_recon (no value/advantage) | PPO + BC (GAE required) |

skip_termination was the root cause of v10 value explosion (3.4M). Switch to ignore_terminations fixed it.

### action_trust Blending Mechanism

`action = trust * model_action + (1-trust) * ref_action`

- `action_trust=0.0` → pure reference PD tracking (BC warmup)
- `action_trust=1.0` → pure model output (PPO training)
- BC warmup optimal: `trust=0.0` + `freeze_encoders=True` + `deterministic_rollout=True`
- Configurable via `SONIC_PHYSX_ACTION_TRUST` env var

### BC Warmup Protocol (v15-validated)

```
SONIC_PHYSX_BC_ONLY=1          // ppo_loss_coef=0, freeze_encoders=True
SONIC_PHYSX_ACTION_TRUST=0.0   // pure ref_action
SONIC_PHYSX_H_THRESH=10.0      // disable height termination
SONIC_PHYSX_IGNORE_TERM=1      // finite episodes, proper GAE
```

This produces policy_avg=0.00067, length=7.84 — the encoder and decoder are correctly adapted to PhysX observations while tracking reference motion.

---

## 2026-08-10 — v13: Encoder-First Two-Step

### v13 Step 1: Encoder Adaptation (PASSED)

| Item | Value |
|------|-------|
| Launch | 2026-08-10 02:01 UTC |
| Log | physx_v13_s1_20260810_020124.log |
| Duration | 200 iterations |
| Checkpoint | SONIC release last.pt |
| SONIC_PHYSX_SKIP_TERM | 1 |
| ppo_loss_coef | 0 |
| g1_recon_coef | 1.0 |
| Iteration time | ~8s |

**Final:** g1_recon=0.0006 @ iter 200. Checkpoint: `physx_ppo_v13_s1-20260810_020155/last.pt`

**Verdict: PASSED.**

### v13 Step 2: BC+PPO from Adapted Encoder (PARTIAL)

| Item | Value |
|------|-------|
| Launch | 2026-08-10 02:29 UTC |
| Log | physx_v13_s2_20260810_022956.log |
| Duration | ~300 iter (killed) |
| Checkpoint | v13 S1 last.pt |
| SONIC_PHYSX_IGNORE_TERM | 1 |
| SONIC_PHYSX_BC_COEF | 1.0 |
| Thresholds | ori=0.6 ank_pos=0.6 ank_h_mult=3.0 |

BC loss stuck at 0.12, length remained at 1.0. Even with correct latent representation (S1 passed), decoder cannot bridge from latent tokens to precise joint actions under PPO.

**Verdict: PARTIAL.** Encoder adaptation works; BC loss after adaptation does not. Problem shifted from "encoder" to "decoder."

---

## 2026-08-10 — v14: g1_dyn Decoder Reinit

**Hypothesis:** The g1_dyn decoder trained on Isaac Sim observations cannot handle PhysX's different numerical range. Solution: reinitialize g1_dyn decoder layers randomly and re-train via BC.

### v14 Sub-experiments

| Run | Description | Result |
|-----|-------------|--------|
| test_best_ckpt | Verify best checkpoint works | BC loss 0.12 stuck |
| test_trust1 | action_trust=1.0 test | length=1.0 (no ref actions) |
| s2a | BC warmup, action_trust=0, freeze encoders | **policy_avg=0.0007, length=5.1** |
| term_diag | What triggers termination | Confirmed h_thresh=0.15m kills at ~9 steps |
| debug | Various debugging runs | — |
| h10 | h_thresh=10.0 test | Survived 500 steps — root cause confirmed |
| v15_test | v15 config dry-run | — |

### v14 s2a: BC Warmup with action_trust=0 (PASSED)

| Item | Value |
|------|-------|
| Launch | 2026-08-10 06:27 UTC |
| Log | physx_v14_s2a_20260810_062702.log |
| Duration | ~300 iter |
| action_trust | 0.0 (100% ref_action) |

**Final:** policy_avg=0.0007, length=5.1 (first time breaking 1.0!) — BC loss near zero.

**Verdict: PASSED.** action_trust=0 + freeze_encoders is the correct BC warmup formula. The g1_dyn reinit hypothesis was wrong. The decoder didn't need reinitialization — it needed action_trust blending to provide a stable behavioral scaffold. With trust=0, the model output is replaced with ref_action, so the encoder learns from correct-answer observations while the decoder trains on pure MSE against ref qpos.

### v14 Post-Mortem

Three routes compared: v14 s2a (action_trust=0 BC) succeeded. v14 g1_dyn reinit route terminated because action_trust blending solved the problem without architectural changes.

---

## 2026-08-10 — v15: BC Warmup Regression Test (PASSED)

**Goal:** Reproduce v14 s2a BC warmup on a clean SONIC release checkpoint, with properly configured thresholds.

**Config:**
```
SONIC_PHYSX_BC_ONLY=1          // ppo_loss_coef=0, freeze_encoders=True
SONIC_PHYSX_ACTION_TRUST=0.0   // 100% ref_action
SONIC_PHYSX_H_THRESH=10.0      // disable height termination
SONIC_PHYSX_IGNORE_TERM=1      // ignore_terminations
```
Thresholds: ori=3.0, ank_pos=3.0, ank_h_mult=10.0 (very loose — exploration-safe)

| Item | Value |
|------|-------|
| Launch | 2026-08-10 09:26 UTC |
| Log | physx_v15_bc_20260810_092647.log |
| Duration | 300 iterations |
| Checkpoint | SONIC release last.pt |
| Model | 37M UniversalTokenModule, 350MB |

**Final metrics:**
| Metric | Value |
|--------|-------|
| policy_avg | **0.00067** |
| length | **7.84** |
| bc_loss | ~0.0001 (MSE ≈ 0) |

**Verdict: PASSED.** Policy perfectly tracks reference motion under BC warmup protocol. Checkpoint: `physx_ppo_v15_bc-20260810_092705/last.pt` (350MB). This is the foundation for v1 production PPO.

---

## 2026-08-11 — v1 Production PPO (COMPLETED at iter 1248)

**Goal:** Full PPO training at 4096 envs from v15 BC checkpoint. PPO learns to generate novel actions surpassing the reference tracking baseline under loose thresholds (ori=0.35, ank=0.35).

**Launch config:** SONIC_PHYSX_ORI_THRESH=0.35, SONIC_PHYSX_ANK_POS_THRESH=0.35, SONIC_PHYSX_ANK_H_MULT=1.5, num_envs=4096, 16 DDP, 1024 workers.

| Item | Value |
|------|-------|
| Launch | 2026-08-11 01:44 UTC |
| Stopped | 2026-08-11 05:10 UTC (3.4h runtime) |
| Log | /tmp/physx_ppo_v1_prod_20260811_014415.log |
| TB | logs_rl/TRL_G1_PhysX/physx_ppo_v1_prod-20260811_014432/ |
| Checkpoints | iter 500, 1000 (428MB each) |
| Model | 37M UniversalTokenModule |
| Scale | 4096 envs × 16 DDP × 1024 workers |
| Throughput | ~11,000 steps/s, ~9s/iter |
| NaN | 0 (clean throughout) |

**Final metrics (iter 1248):**
| Metric | iter 250 | iter 500 | iter 1000 | iter 1248 |
|--------|----------|----------|-----------|-----------|
| Rewards | 7.85 | 8.66 | 17.60 | **35.37** |
| Mean length | 4.33 | 5.31 | 12.88 | **26.59** |
| Per-step reward | 1.81 | 1.63 | 1.36 | **1.33** |
| Entropy | -39.05 | -34.0 | -29.6 | **-27.5** |
| Noise std | 0.08 | 0.09 | 0.10 | **0.11** |

**Key insight:** Per-step tracking reward stabilized at ~1.33 after iter 1000, meaning tracking quality reached a floor. Further length gains (13→27) came purely from the policy learning to avoid termination, not from better motion tracking. The loose thresholds (ori=0.35) were no longer providing useful training signal at this point.

**Verdict: COMPLETED.** Length 26.6 = 3.4× BC baseline (7.84). Training healthy throughout, zero NaN. Stopped at iter 1248 to begin threshold tightening phase.

**Checkpoint:** `model_step_001000.pt` used to launch v2 tight.

---

## 2026-08-11 — v2 Threshold Tightening (RUNNING)

**Goal:** Tighten thresholds from ori=0.35→0.30, ank=0.35→0.30, continue training from v1 iter 1000 checkpoint. The relaxed thresholds in v1 allowed the policy to reach length >20 but tracking quality plateaued. Tighter thresholds force the policy to improve motion tracking or terminate earlier.

**Config:**
```
SONIC_PHYSX_ORI_THRESH=0.30     // tightened from 0.35
SONIC_PHYSX_ANK_POS_THRESH=0.30 // tightened from 0.35
SONIC_PHYSX_ANK_H_MULT=1.5      // unchanged
SONIC_PHYSX_H_THRESH=10.0       // unchanged (off)
SONIC_PHYSX_IGNORE_TERM=1
```

| Item | Value |
|------|-------|
| Launch | 2026-08-11 05:10 UTC |
| Log | /tmp/physx_ppo_v2_tight_20260811_051058.log |
| Checkpoint | v1 prod iter 1000 |
| Schedule | 10000 iterations, model save every 500 |
| NaN | 0 (so far) |

**Plan:** v2 (0.30) → v3 (0.25) → v4 (0.20 Isaac standard) → evaluate on Isaac Sim.

---

## 2026-08-11 (afternoon) — v3: Root Velocity Fix & NaN Guard ⭐

### Background

After container restart, all attempts to resume or start training failed with length degrading from ~2.6→1.0 within 3 iterations. Extensive diagnostic narrowed it to TWO independent root causes in the environment, not in the model or checkpoints.

### Root Cause 1: Root Velocity Not Reset on env.reset()

**Discovery process:**
1. action_trust=0 + BC_ONLY still degraded (2.59→1.23→1.02→1.0): NaN guard alone insufficient
2. Standalone env test: 100 episodes, action_trust=0, first 20 mean=2.35, last 20 mean=1.00 → env itself degrades
3. Reset verifies: joint positions PERFECT after reset, root pose PERFECT, but `get_root_world_velocity()` = z=-14 m/s
4. PhysX `setRootGlobalPose()` teleports position/orientation but DOES NOT clear linear/angular velocity
5. After termination (robot falls at 1.4g), residual downward velocity persists
6. Next step integrates 14 m/s * 0.02s = 0.28m downward → robot teleports through ground → immediate termination

**Fix:**
- Added `set_root_world_velocity(lin, ang)` to C++ bindings (`physx_bindings.cpp`, 3 locations: declaration + impl + pybind11 def)
- Impl: `ptr->setRootLinearVelocity(np_to_v3(lin), true); ptr->setRootAngularVelocity(np_to_v3(ang), true);`
- Call in `PhysXEnv.reset()`: `self.art.set_root_world_velocity(zeros(3), zeros(3))`
- Rebuilt `.so` on NPU: `cd build && make -j4`

**Before/After:**
| | Before | After |
|---|---|---|
| 100-ep standalone test | 2.35→1.00 (degrades) | **9.00 stable across ALL 100 eps** |
| 4096-env training iter 1 | length=2.59 | **length=9.54** |
| 4096-env training iter 6+ | length=1.00 | **length=9.2, stable** |

### Root Cause 2: NaN Through IEEE 754 Action Blending

**IEEE 754 trap:** `0.0 * NaN = NaN` (not 0.0)

```
action = trust * model_action + (1-trust) * ref_action
       = 0.0 * NaN + 1.0 * ref_action
       = NaN                ← IEEE 754, not ref_action!
```

When model produces NaN (random weights post-PPO update, or corrupted checkpoint), NaN leaks through PD targets → physics engine → all observations → SHM → all environments.

**Fix:** `np.isfinite()` guard in `PhysXEnv.step()` L549-552:
```python
if not np.isfinite(action).all():
    action = ref_action.copy()
else:
    action = self.action_trust * action + (1.0 - self.action_trust) * ref_action
```

### Files Modified

| File | Change |
|------|--------|
| `physx_bindings.cpp` | +`set_root_world_velocity()` (3 additions) |
| `physx_env.py` | root vel reset L523-524 + NaN guard L549-552 |
| `train_agent_trl.py` | +`SONIC_PHYSX_BC_CHECKPOINT` env var loading L626-636 |

### Verification (v3 verify)

**Config:** BC_ONLY=1, action_trust=0, noise_ctl=0, 4096 envs, 10 iter
**Result:**
| Iter | Length | Reward |
|------|--------|--------|
| 1 | 9.54 | 48.7 |
| 2 | 9.44 | 48.8 |
| 3 | 9.06 | 46.9 |
| 4-10 | ~9.2 | ~48 |
| NaN | **0** | |

**Verdict: PASSED.** 4096-env infrastructure confirmed stable after root velocity fix. Episode length 3.7× pre-fix baseline (2.5).

---

## 2026-08-11 (afternoon) — v4: BC Warmup from Phase B Checkpoint

### Context

Phase B checkpoint (`/tmp/phase_b_checkpoint.pt`, 3.2MB) was from 2026-08-07 PPO 64-iter verification (standing). Contains `policy` + `bc_actor` state dicts. Loading this provides a policy that already learned to stand under PhysX physics.

### Config

```
SONIC_PHYSX_BC_COEF=1.0          // BC loss enabled: MSE(action_mean, ref_qpos)
SONIC_PHYSX_ACTION_TRUST=0.0     // environment uses pure ref_action
SONIC_PHYSX_BC_CHECKPOINT=/tmp/phase_b_checkpoint.pt  // new: load previous policy
SONIC_PHYSX_IGNORE_TERM=1        // finite episodes, proper GAE
SONIC_PHYSX_NOISE_CTL=0          // no noise clamping
```

### Code Change (train_agent_trl.py)

Added `SONIC_PHYSX_BC_CHECKPOINT` env var trigger (L626-636). Loads policy weights with `strict=False` before the existing `pretrained_model` block. Phase B ckpt had missing=55, unexpected=13 keys — expected due to config differences between Phase B and current run.

### Current Status

| Item | Value |
|------|-------|
| Launch | 2026-08-11 ~12:04 UTC |
| Log | `/tmp/physx_bc_20260811_120443.log` |
| Duration | ~5/100 iter (so far) |
| Checkpoint | Phase B `/tmp/phase_b_checkpoint.pt` |
| action_trust | 0.0 |
| BC coef | 1.0 |
| Model | 37M UniversalTokenModule |

**Early metrics (iter 1-5):**

| Iter | Length | Reward |
|------|--------|--------|
| 1 | 9.55 | 48.7 |
| 2 | 9.45 | 48.8 |
| 3 | 9.07 | 46.9 |
| 4 | 9.32 | 48.2 |
| 5 | 9.24 | 48.3 |

**Assessment:** Length stable ~9.2-9.5, no degradation, 0 NaN. BC loss active — model is learning to predict reference actions from observations. At action_trust=0 the model output is discarded for physics but the BC gradient flows through the decoder.

### Next Steps After v4

1. BC run to 100 iter → save checkpoint
2. action_trust=0 → 0.5 → 1.0 gradual transition (model's own actions start to drive physics)
3. At trust=1.0 + stable length → enable PPO (BC_COEF=0, add noise_ctl)
4. Threshold tightening: ori=0.35 → 0.30 → 0.20

---

## Appendix: NPU Server Info

| Item | Value |
|------|-------|
| IP | 113.46.41.54 |
| Container | sonic-train |
| Code path | /root/GR00T-WholeBodyControl |
| Logs path | /root/GR00T-WholeBodyControl/logs_rl/TRL_G1_PhysX/ |
| Disk | 50GB overlay, ~62% used |
| RAM | 2 TB |
| GPU | Ascend 910B × 16 |

---

## 2026-08-12 — Trust Transition + Mixed PPO 系列实验 (v2-v8)

### 背景

BC warmup 完成后 (len≈9.6)，通过渐进 trust transition (0→0.1→0.2→0.3→0.4→0.5) 让模型逐步接管控制权，最后切入混合 PPO (BC loss + PPO 同时在线)。

### Trust Transition 结果

| 阶段 | 起点 | 终点 | 结果 |
|------|------|------|------|
| 0→0.1 | BC warmup | len≈7.5 | 平滑 |
| 0.1→0.2 | trust=0.1 | len≈9.5 | 平滑恢复 |
| 0.2→0.3 | trust=0.2 | len≈9.6 | 平滑 |
| 0.3→0.5 (跳过0.4) | trust=0.3 | len=1.56 | **崩溃** — 0.2 跳跃过大 |
| 0.3→0.4 | trust=0.3 | len≈7.5 | 可接受 |
| 0.4→0.5 | trust=0.4 | len=4.4 恢复中 | 边界情况 |

**教训**: trust 每次最多 +0.1，≥0.5 后 BC-only transition 收益递减，应切 PPO。

### Mixed PPO 实验矩阵

| 版本 | BC_COEF | BC 参考 | 起点 | 结果 |
|------|---------|---------|------|------|
| v1 | 1.0 | trust=0.5 | trust=0.5 | 失败: len=2.09, 恢复太慢 (noise 0.10 太大) |
| v2 | 2.0 | trust=0.5 | trust=0.5 | len=12.82 @ iter 183 后崩溃 (5.29→12.82) |
| v3 | 2.0 | **v2 last.pt (自身)** | v2 iter 183 | 失败: 横盘 10.5 (BC 梯度=0) |
| v4 | 1.0 | **v2 last.pt (自身)** | v2 iter 183 | 失败: 横盘 10.5 |
| v5 | 0 (纯PPO) | v2 last.pt | v2 iter 183 | 失败: 横盘 10.2-10.5, TB crash |
| **v6** | **1.0** | **trust=0.5** | **trust=0.5** | **✅ len=15.43 @ 300 iter 完整完成** |
| v7 | 0 (纯PPO) | v6 last.pt | v6 iter 300 | 失败: 横盘 15.2-15.6 (46 iter) |
| v8 | 0.5 | trust=0.5 | trust=0.5 | **RUNNING** — iter 21 时 len=9.03 vs v6 同期 8.54 |

### 关键发现: BC 参考必须 ≠ 模型权重

v3/v4/v5 全部横盘的原因: BC_CHECKPOINT 同时用作模型初始化 + BC 参考。当 BC 参考 = 模型自身权重时，BC loss 梯度恒为 0（模型输出已经匹配参考），BC 约束失效。

**v2/v6 成功的原因**: BC 参考是 trust=0.5 模型（输出不同于 trust=1.0 的模型），BC loss 提供 productive tension——推动模型调整 + PPO 奖励引导方向。

### v6 完整轨迹 (300 iter)

| 阶段 | Iter | Len | Reward |
|------|------|-----|--------|
| 起步 | 1 | 5.31 | 12.3 |
| BC 恢复 | 20 | 8.50 | 20.6 |
| 突破 10 | 48 | 10.05 | 24.9 |
| 稳步攀升 | 100 | ~11.7 | ~30.5 |
| | 150 | ~12.5 | ~33 |
| | 200 | ~13.5 | ~35 |
| | 250 | ~14.4 | ~37 |
| 最终 | **300** | **15.43** | **39.0** |

### v7 教训: 纯 PPO 从 checkpoint 继续会横盘

v7 从 v6 len=15.43 继续，BC_COEF=0。iter 1 len=16.42 (BC 解除瞬间+1)，随后 46 轮横盘 15.2-15.6。纯 PPO 稀疏奖励 + 新 optimizer 从零建动量 → 无法突破。

### v8 假设: BC_COEF=0.5 是甜蜜点

v6 (1.0) 到 15.43，v7 (0) 横盘。0.5 介于中间——保留 BC tension 但减少回拉。初期数据支持: iter 21 len=9.03 vs v6 同期 8.54 (+0.5)。

### 磁盘教训

- wandb import 在 /tmp 无空间时崩溃 (OSError: No space left on device)
- Docker overlay COW: 删除文件释放空间有限
- 已清理 v8-v15, v1_prod, v2_phase/tight/snap 等旧目录，释放 ~10GB

### 下一步

1. v8 完成后对比 BC_COEF ∈ {0.5, 1.0, 2.0} 最优值
2. 阈值收紧: ori=0.35 → 0.30 → 0.20 (Isaac 标准)
3. BC_COEF 递减方案: 从最优值逐步降到 0，让 PPO 完全接管

---

## 2026-08-13 — 重大修正: BC loss 真实机制

### 错误理论 (2026-08-12)

之前认为 "BC_CHECKPOINT 同时用作模型初始化 + BC 参考，BC 参考必须 ≠ 模型权重才有 productive tension"。**这是错的。**

### 真相 (读 ppo_trainer_aux_loss.py:190-192)

```python
if self.compute_bc_loss:
    action_mean = forward_results["policy_results"]["action_mean"]
    ref_action = mb_rollout_data["mb_obs_dict"]["ref_action"]  # ← 动捕数据！
    bc_loss = mse_loss(action_mean, ref_action)
```

- **BC loss = MSE(模型动作, reference motion 的 qpos)** — 参考是**动捕数据**，永远不变，与任何 checkpoint 无关
- **BC_CHECKPOINT 只做一件事: 初始化模型权重**
- trainer 是 `TRLAuxLossPPOTrainer` (trl_ppo_aux.yaml)，不是 ppo_trainer.py

### 真实机制

| 实验 | 初始化权重 | iter 1 len | 结果 | 解释 |
|------|-----------|-----------|------|------|
| v6/v8 | trust=0.5 ckpt (差策略) | 5.31 | 爬到 15.4 | advantage 大 → PPO 强信号 |
| v3/v4/v5 | v2 last.pt (好策略) | 10.4-10.5 | 横盘 | advantage 小 → PPO 无梯度 |
| v7 | v6 last.pt (好策略) | 16.4→15.3 | 横盘 | 同上 |

**结论**: BC loss 是防崩塌锚（拉向动捕），上升动力 100% 来自 PPO advantage。**从差策略起步才有学习信号**——这是 v6/v8 成功的真正原因。BC_COEF ∈ {0.5, 1.0} 等效（都是够用的安全网）。

### 对阈值收紧的影响

阈值收紧实验 (ori=0.30) 从 trust=0.5 起步 len=3.29——差策略 + 更严阈值 = 更大的 advantage 空间，理论上能爬到更严阈值下的新最优。

### v8 最终结果

| 版本 | BC_COEF | Final len @ 300 iter |
|------|---------|---------------------|
| v6 | 1.0 | 15.43 |
| v8 | 0.5 | ~15.10 |
| 结论 | | 等效，BC_COEF 非关键变量 |


---

## 2026-08-13 — 阈值收紧: ori=0.35→0.30 完成

### 训练配置

与 v6 完全相同的配方（trust_0_5 ckpt 起步 + BC_COEF=1.0 + noise 0.03/0.05），仅阈值收紧:
- ori_thresh: 0.35 → **0.30**
- ank_pos_thresh: 0.35 → **0.30**
- ank_h_mult: 1.5 (不变)
- exp dir: `physx_ppo_threshold_030-20260812_192852`

### 轨迹 (iter → len)

| iter | v6 (0.35) | threshold_030 (0.30) | 差距 |
|------|-----------|---------------------|------|
| 1 | 5.31 | 3.29 | -2.0 |
| 50 | 10.05 | 8.65 | -1.4 |
| 100 | 11.7 | 10.12 | -1.6 |
| 150 | 12.6 | 11.10 | -1.5 |
| 200 | 13.5 | 12.01 | -1.5 |
| 250 | 14.4 | 12.58 | -1.8 |
| 300 | 15.43 | 13.33 | -2.1 |

Final: **len=13.33, reward=34.8**（300 iter）。爬升未饱和（最后 25 iter 仍在 +0.75）。

### 交叉评测 (physx_cross_eval.py)

新增 `scripts/physx_cross_eval.py`: 单 env 多 episode (12×500 步) 确定性 rollout，任意阈值下评测任意 checkpoint。待 v6@0.30 与 t030@0.30 结果对比。

### 交叉评测 2×2 矩阵结果 (24 ep, 匹配动捕种子)

| 策略 | 训练阈值 | eval @ 0.30 | eval @ 0.35 |
|------|---------|-------------|-------------|
| v6 | 0.35 | **22.33** | 22.67 |
| t030 | 0.30 | 15.00 | 14.29 |

**结论: 阈值收紧训练适得其反。** v6 (宽松训练) 在 0.30 严格阈值下 len=22.33，几乎阈值不敏感；t030 (严格训练) 即使回到 0.35 也只有 14.29，且低于自身 0.30 训练长度。

### 机制解释

与 "从差策略起步才有 advantage" 同源：严格终止条件把训练 episode 截断在 ~13 步，策略只学到短期存活，永远看不到长视界行走的动力学；宽松阈值 (0.35) 让 episode 跑到 15-20 步，策略学会完整步态后对阈值自然鲁棒。

**教训**: 阈值收紧不是训练杠杆，是评测杠杆。训练保持宽松阈值 (0.35)，评测逐步收紧到 Isaac 标准 (0.30→0.20)。v6@0.20 评测进行中。

### 修正后路线

1. ~~阈值收紧训练~~ (已证伪) → 训练始终 0.35
2. 从 trust_0_5 ckpt 起步延长训练 (v6 在 300 iter 仍在爬升 14.4→15.43)，500-600 iter
3. 评测按 0.30 / 0.20 阈值报告

### v6 @ Isaac 全标准 (0.20) 评测

**len=18.38** (median=17, min=1, max=45, reward=64.8)。宽松训练的策略在 Isaac 最严阈值下仍有 17 步中位生存；min=1 表明部分困难动捕片段在 0.20 下立即失败。

### 延长训练启动 (2026-08-13 01:20 UTC)

v6 配方 500 iter (exp_var=ppo_500iter): trust_0_5 起步 + BC_COEF=1.0 + ori/ank=0.35。依据: v6 在 300 iter 时仍在爬升 (14.4→15.43)，且宽松阈值训练+严格评测路线已确立。log: /tmp/physx_ppo_500iter_20260813_012006.log

### 性能诊断: 1 iter/min 根因 + 128 workers 修复 (2026-08-13 03:00 UTC)

用户问 "为什么 1 iter/min，envs 并发正确吗？" 诊断结果:

**并发正确**: 51 进程 = 16 accelerate trainer + 32 workers + 3 辅助。4096 envs = 16 × 256 envs，每 trainer 2 workers × 128 envs。全部满载。

**根因: barrier 关键路径串行**。日志官方耗时分解: Collection 49.7s (89%) + Learning 6.2s。每 rollout 步所有 worker barrier 等齐，时延 = 最慢 worker 顺序跑 128 envs。实测 env.step ≈ 16ms (10 物理子步 + Python FK + 4336 维 obs) → 128 × 16ms ≈ 2s/步 × 24 步 ≈ 49s。机器 320 核，CPU 90% 空闲，load 仅 50。

**修复: SONIC_PHYSX_WORKERS 32 → 128** (每 worker 32 envs)。零代码改动:
- Collection 49.7s → 13.9s，吞吐 1760 → 5404 steps/s (3.1×)
- Iteration time ~56s → 17.6s
- 500 iter 预计 6.7h → 2.8h

### Checkpoint 保存崩溃 (02:52 UTC)

iter 50 首次保存时 rank 0 崩溃: `Failed to save checkpoint after 5 attempts. Error: [enforce fail at inline_container.cc:659] unexpected pos 349524544 vs 349524432` → ChildFailedError → 全 run 死亡。

**根因分析**: torch stream writer 短写 (short write) — 预期位置比实际多 112 字节。容器内当时 6.4G 空闲、checkpoint ~450MB，容器内 ENOSPC 不可能。但宿主机 sda2 92% 满 (23G free)，共享 K8s 节点，崩溃时刻宿主 load=64 (外部租户负载)。**最可能: 其他租户瞬时填满宿主磁盘 → overlay 写失败 → 短写**。5 次重试全败 (持续 >25s) 支持瞬时/持续磁盘压力说。

**缓解**: 清理 /tmp 再生性 BC checkpoint (~400MB) + 用户批准删除被杀旧 run 3 目录 (~450MB) → 7.6G 空闲。重启 128-worker 训练 (05:11 UTC)。

### 保存失败模式升级 (06:00 UTC 观察)

重启后（128 workers）保存失败仍在，且呈决定性模式:

| 保存点 | model_step | last.pt |
|--------|-----------|---------|
| iter 50 | ✅ | ✅ |
| iter 100 | ✅ | ❌ 5次重试全败 |
| iter 150 | ❌ 5次重试全败 | ❌ 5次重试全败 |

**决定性特征**: iter 100 的 last.pt 和 iter 150 的 model_step 失败在**完全相同的字节位置** (103532992 vs 103532880，差 112 字节)。同结构 checkpoint → torch 缓冲 flush 在确定性偏移 → 宿主磁盘压满时刻的短写。iter 150 第二次失败 "pos 64 vs 0" = 文件创建后连头部都没写进去（磁盘完全满）。

**结论 (后被修正)**: 宿主 sda2 (92% 满) 被共享租户周期性压满，我们的保存窗口 (~10s) 撞上就失败。训练本身不受影响 (16.5s/iter 继续)。**⚠️ 真实根因见下一节 — 是 Docker backing store /mnt/paas 100% 满，与 sda2 无关。**

**防御**: 
1. run 存活确认 — 重试逻辑扛住了（与首次崩溃不同，原因待查）
2. checkpoint 备份到宿主 /tmp (tmpfs 973G 空闲)，每 5 分钟 docker cp 刷新
3. 已落盘: iter 50 (完整) + iter 100 (model_step)。最坏情况从 iter 100 restart

### 磁盘满真实根因: /mnt/paas 共享 backing store (06:10-06:30 UTC 发现)

之前归因于"宿主 sda2 92% 满"是**错的**。实证链:

1. **dd 测试**: 容器内 `dd if=/dev/zero of=test bs=1M count=500` 静默只写了 112MB (107 records) — 写缓冲在 page cache，flush 到满盘时静默截断，无 ENOSPC 报错
2. **docker info**: Docker Root Dir = `/mnt/paas/runtime`，挂载 `/dev/mapper/vgpaas-share` (1.2T) — **共享 K8s 池被其他租户写满 100%**，与我们无关
3. **容器内 df 是合成的** — overlay 报告的"空闲"不反映 backing store 真实容量；写入 ~100MB page cache 缓冲后 flush 失败，表现为**确定性偏移的短写** ("unexpected pos"，112 字节差)

**缓解**: experiment dir 移到 `/dev/shm` (tmpfs 16G，容器内独立) + symlink；BC ref checkpoint 也放 /dev/shm；宿主侧备份到宿主 /tmp (tmpfs 973G，非 K8s 池) 每 10 分钟 tar 刷新。**tmpfs 数据容器重启即失，训练完成后必须移到持久存储。**

### Resume 崩溃根因: hydra 配置 + 容器 /tmp 隔离 (06:35 UTC)

500 iter 延长训练在 iter 150 后 checkpoint 全败、磁盘 (Docker backing store /mnt/paas) 100% 满后，重启走 resume 路线: experiment dir 移到 /dev/shm (tmpfs 16G) + symlink。但 resume 启动即崩 (~30s, exitcode 1, 无 traceback, hydra output dir 未创建)。

**真根因 (两个叠加)**:
1. **hydra 配置错误**: config 已定义 `experiment_dir` (sonic_release.yaml:25)，`+experiment_dir=...` 报 "Could not append to config. An item is already at 'experiment_dir'" → 所有 rank 在 config 解析阶段退出。修复: `++experiment_dir=...` (override)。
2. **容器 /tmp 与宿主 /tmp 不共享**: scp 到宿主 /tmp 的修复版脚本，容器里 `docker exec sh /tmp/...` 读的是容器自己的旧版 /tmp。必须 `docker cp` 进容器。

**教训**: 之前所有 "写满 overlay → hydra 建目录失败" 假设都是错的 — crash 发生在 hydra 解析 config 阶段，与磁盘无关。saved meta 出现 = 通过该阶段。resume 训练 06:35 启动成功。

### Resume 第二次崩溃: torch_npu relanding 破坏 deepcopy (06:50-07:00 UTC)

resume 训练正常运行 50 iter 后在 iter 150 保存时崩溃 — 但失败点完全不同: `copy.deepcopy(state)` (model_save_callback.py:123) 抛 `RuntimeError: Expected a Storage of type float or an UntypedStorage, but got type Byte`。

**根因链 (NPU 上逐步实证)**:
1. trainer load 用 `torch.load(ckpt, map_location=npu)` → torch_npu 对老格式权重做 "relanding" 转换 → 产生的 tensor dtype=float32、storage_dtype 也报告 float32，但 `__deepcopy__` 时内部 `_typed_storage()` 返回 Byte storage → set_ 失败
2. 原 run 里这两个 tensor (`state.cur_reward_sum`/`cur_episode_length`, 每 trainer 256 env 的 per-env 奖励统计) 是 `torch.zeros(device=npu)` 新建的 → 存储干净 → deepcopy 正常。resume 后被 relanded tensor 替换 → 首次保存必崩
3. checkpoint 里存的 tensor 本身是干净的 (CPU load 验证 storage=float32, deepcopy OK) — 坏在 NPU map_location 转换这一步

**修复**: ppo_trainer.py load state 段, cur_reward_sum/cur_episode_length 赋值前用 `torch.empty(shape, dtype, device).copy_(value)` 重建干净存储。NPU 上验证: 修复后 state deepcopy OK。

**教训**: NPU 上 torch.load 的 tensor 不保证能 deepcopy — 任何 checkpoint→内存→再保存的链路都要警惕 relanding。诊断手段: load 后逐个字段 deepcopy 定位。

### 500 iter 延长训练完成 (09:00 UTC)

第三次 resume 后一路跑完 500 iter，**16 次保存零失败**（relanding 修复生效）。全部 checkpoint (50-500) + last.pt 在 /dev/shm (4.7G/16G)，宿主 tmpfs 备份同步 (子目录 physx_ppo_ppo_500iter-20260813_051226/)。

**训练指标**: Mean length @0.35 从 iter 100 的 ~10.3 爬升到 iter 500 的 **18.8-18.9**（v6 在 300 iter 是 15.43，延长训练继续 +3.4）。iter 150 后 ~17 起步，400-500 区间 18.7-19.1 稳步爬升未饱和。

**三阈值交叉评测**: 最终 checkpoint model_step_000500.pt，0.35/0.30/0.20 × 24 episodes × motion_seed 0，结果见下。

### 交叉评测结果 (完成, ~10:00 UTC)

| 阈值 | mean_len | median | min | max | mean_rew | vs v6 (300 iter) |
|------|----------|--------|-----|-----|----------|------------------|
| 0.35 | 27.08 | 24.5 | 11 | 70 | 85.8 | 22.67 → **+4.4** |
| 0.30 | 26.79 | 24.5 | 11 | 70 | 85.1 | 22.33 → **+4.5** |
| 0.20 | 29.50 | 29.0 | 7 | 57 | 92.8 | 18.38 → **+11.1 (+60%)** |

**评测结论**: 500 iter 策略在 0.35/0.30/0.20 三阈值下 mean_len = 27.08/26.79/29.50 — 阈值不敏感，且在 Isaac 全标准 (0.20) 下比 v6 (18.38) 提升 +60%。延长训练价值确认。

**训练 18.8 vs 评测 27.1 的差异解释**: 训练 mean length 是全部 4096 envs 带探索噪声 (noise_std=0.05) 的均值；评测是单 env + 确定性推理 (act_inference 无采样) + 匹配动捕种子。确定性推理必然优于带噪均值，两者不可直接比较。

**持久化**: model_step_000500.pt + config.yaml 下载到本地 `checkpoints_backup/`，**MD5 校验通过** (本地 = 容器 = `9139d232be0050626923fe75a51bf57b`)。容器 /dev/shm 保留全部 11 个 checkpoint (50-500) 供后续评测。宿主 tmpfs 备份完整。

### 500 iter 策略渲染验证 (10:20-10:30 UTC)

**方法**: NPU 上 `record_ppo500.py` 用 model_step_000500.pt 跑单 env 确定性 rollout（skip_termination=True → 不内部 reset、不截断，500 步连续轨迹）→ npz 下载本地 → MuJoCo OSMesa 渲染 MP4 (`ppo500_full.mp4` 全程 + `ppo500_gait.mp4` 前 150 步) + 9 张关键帧。本会话无法直接查看图像，采用定量指标 + 用户肉眼双验证。

**定量结果**:
1. **存活**: 本 episode 52 步才突破 |dz|>15cm（评测均值 27）— 高于均值的一集。前 30 步 dz max 仅 51mm，行走健康。50 步前进 1.18m（≈1.2 m/s，动捕步频的快步走）
2. **关节振荡 (Phase C 关注点)**: 踝关节高频分量 peak-to-peak 93-173 mrad (5-10°)，**低于** ref PD 基线 292-403 mrad (17-23°) 和 Phase B 旧策略 343-529 mrad — 无振荡回退，策略比 PD 基线更干净
3. **失败模式**: ~52 步后下沉 (dz -74mm) → PD 补偿失稳 → **弹射起飞** (step 100 dz=+1.1m, step 500 达 +6.3m)。这是 ignore_terminations 训练下策略进入未训练状态域后的已知崩溃模式（termination 已触发但被忽略 → 动作退化 → PD 力矩累积），非物理 bug

**结论**: 步态自然、无振荡回退；出域崩溃模式与预期一致。视频待肉眼确认。渲染产物在本地 job tmp (ppo500_full.mp4 / ppo500_gait.mp4 / ppo500_f*.png)，轨迹数据 ppo500_trajectory.npz。

### 1000 iter 延长训练完成 (~12:50 UTC)

500→1000 iter resume，**20 次保存零失败**（relanding 修复持续生效）。最终 mean length **23.2-24.0 @0.35**（iter 500 的 18.8 → **+4.7**）。resume 曲线从 19.1 稳步爬升至 24.0，全程无饱和——延长训练还有空间。

**三阈值交叉评测** (model_step_001000.pt, 24 eps, motion_seed 0):

| 阈值 | mean_len | median | vs 500-iter | vs v6 (300 iter) |
|------|----------|--------|-------------|------------------|
| 0.35 | **32.21** | 31 | 27.08 → +5.1 | 22.67 → +9.5 |
| 0.30 | **32.08** | 31 | 26.79 → +5.3 | 22.33 → +9.8 |
| 0.20 | **30.88** | 31 | 29.50 → +1.4 | 18.38 → **+12.5 (+68%)** |

**关键发现**: 0.35/0.30 阈值下 +5 大幅提升（策略更好地保持在参考姿态附近），但 0.20 严格阈值只 +1.4——三阈值 median/min/max 完全一致 (31/14/56)，说明**终末事件是同一个**（跌倒），与阈值无关。500→1000 iter 把跌倒时间从中位 24.5 推迟到 31，但严格阈值下接近边际收益递减。下一步若要继续突破 0.20 存活，需要针对跌倒模式本身（而非阈值容差）做工作。

### 跌倒模式诊断 (13:00-13:30 UTC)

**方法**: monkey-patch `_check_termination` 捕获终止原因（不动容器代码），48 集 × 2 motion seeds @ 0.20 严格阈值，记录终止前 10 步误差曲线。对照: 纯 ref PD（无前瞻）同条件 48 集。

**终止原因分布 (1000-iter 策略)**:
| 原因 | 占比 |
|------|------|
| height (dh>0.15) | **75%** |
| ankle_pos | 15% |
| ori | 6% |
| wrist_h | 4% |

**失败动力学 (渐变，非突变)**: 终止前 10 步 dh 从 55mm 线性爬升到 129mm。策略在慢慢"蹲下去"，不是突然摔倒。

**PD 对照 (决定性)**: 纯 ref PD mean_len=**9.21**（每集 8-10 步），**100% 死于 height**，死亡时 dh 全部在 0.150-0.185 窄带。策略 (31 步) 是纯 PD 的 3.4 倍 — token 前瞻在起作用，但物理层有系统性根高下沉。

**机制假设**: mocap 根轨迹与 G1 质量属性物理不一致 → 机器人系统性偏低 (dz 均值 -20~-30mm + 步频振荡 ±30-60mm) → 15cm 预算被吃掉 ~1/5 起步，PD 无根关节直接驱动，无法纠偏。验证实验: leg kp × offset 扫描 (ref PD 存活 @ 严格阈值)。

**已知奖励缺陷**: r1 (root tracking) = 0.5×exp(-dh²/0.09)，dh=150mm 时仍给 0.194 (满分的 39%) — 对高度误差几乎不敏感。若物理扫描确认根高可纠，下一步奖励塑形：加陡高度项 + 从 1000-iter checkpoint 续训。

### 物理杠杆扫描结论 + Height-hinge 重训启动 (13:33 UTC)

**物理杠杆全部排除** (ref PD 存活 @0.20 严格阈值, 48 eps):
| 杠杆 | 结果 |
|------|------|
| leg kp ×1.5 / ×2.0 | 9.15-9.21 (无变化) |
| root_z_offset 0 / -0.02 / -0.04 / -0.06 | 9.15-9.17 (无变化) |
| 瞬态追踪 | robot root 单调自由落体 vz -1~-2 m/s，ref root 几乎静止 (mocap 帧间 std 2.4mm) |
| 重置姿态 FK | 双脚悬空 +22~36mm |
| skip_termination 自由跑 | root_z 0.75→0.09 击穿地面, foot_z→NaN |

**根因确认**: mocap 根轨迹在 PhysX 物理下不是有效前馈 — 重置即悬空自由落体，ref PD 9.2 步死亡是物理必然。策略的 31 步是学习到的补偿 (3.4× PD)，但基础前馈的根高下沉无法用 PD 增益或 offset 修正。**物理层无更多杠杆，转向策略层：奖励塑形。**

**干预: height hinge 奖励**
- 代码: `physx_env.py` `height_hinge_weight` (config key, 默认 0.0 不影响评测脚本) + `train_agent_trl.py` `SONIC_PHYSX_HEIGHT_HINGE` 环境变量
- 奖励项: `+w × max(0, 1 - |dh|/h_thresh)`，h_thresh 与终止一致 (站立 0.15 / 蹲 0.75)。dh=0 满分，线性衰减到终止阈值恰好归零 — 梯度恒定 (w/0.15 = 13.3/m @ w=2)，而 r1 高斯在 dh=0 处梯度为零、dh=150mm 仍付 39%
- Smoke test: hinge=2.0 时 dh=2mm 加 +2.23、dh=38mm 加 +2.02，梯度正确
- **launch (13:33)**: in-place resume model_step_001000.pt，1000→1300 iter，v6 配方 + `SONIC_PHYSX_HEIGHT_HINGE=2.0`，log /tmp/physx_hinge_20260813_133259.log
- 预期: 训练时奖励均值 +~1.5-2.0 (hinge 常态贡献)，critic 需数 iter 适应新标度；300 iter 后 0.20 严格阈值交叉评测对照 30.88 基线

### Reset-drop 诊断 (13:55 UTC, hinge 训练并行)

假设: 重置悬空 (双脚 +22~36mm) 的自由落体瞬态是 PD 9.2 步死亡的主因 → 把**机器人** root 下移到脚触地（此前 offset 扫描只动参考）。

| drop | robot_only (dh 起始=drop) | both (dh 起始=0) |
|------|---------------------------|-------------------|
| 0.02 | 8.73 (dh@death=20mm) | 9.21 |
| 0.03 | 8.23 (dh@death=30mm) | 9.17 |
| 0.04 | 7.92 (dh@death=40mm) | 9.17 |
| 基线 | 9.15 (dh@death≈150-185mm) | 9.21 |

**决定性**: robot_only 下机器人不再下落 (dh 恒定=drop，支撑成立)，但 8-9 步照死——死亡原因变成 height 以外的项。both 与基线完全一致。**~9 步是 PD 对该动捕+增益的固有跟踪极限**，自由落体只决定哪个终止条件先触发 (height)，不决定何时死亡。reset-drop 杠杆排除，物理层彻底到顶，奖励塑形 (hinge) 是唯一方向。

### Hinge 重训完成 + 评测大突破 (15:02 UTC)

1000→1300 iter, v6 配方 + SONIC_PHYSX_HEIGHT_HINGE=2.0，全程无 NaN，保存零失败。训练内 mean length 24.9-25.8@0.35 (1000-iter 末段 23.2-24.0)。

**三阈值交叉评测 (model_step_001300.pt, 24 eps, motion_seed 0)**:

| 阈值 | 1000-iter 基线 | Hinge 1300 | Δ |
|------|---------------|------------|---|
| 0.35 | 32.21 | **40.29** | **+8.1 (+25%)** |
| 0.30 | 32.08 | **41.00** | **+8.9 (+28%)** |
| 0.20 | 30.88 | **37.92** | **+7.0 (+23%)** |

**决定性证据**: 0.20 严格阈值 +7.0（500→1000 延长训练只 +1.4）。训练长度延长已边际递减，hinge 是首个突破 0.20 平顶的杠杆——印证诊断: 策略此前对高度下沉不敏感 (r1 高斯 dh=0 零梯度、dh=150mm 付 39%)。恒定梯度奖励让策略学会主动保持根高。

### Hinge 1300 渲染验证 (08-14)

方法同 500/1000: `record_ppo1300.py` skip_termination 连续轨迹 500 步 (同一 motion 顺序, RandomState(0)) → npz → 本地 MuJoCo OSMesa `render_ppo1300.py` → ppo1300_full.mp4 / ppo1300_gait.mp4 + 9 关键帧。

**三代连续模式 dz 对比（同动捕种子）**:

| 指标 | 500-iter | 1000-iter | 1300 hinge |
|------|----------|-----------|------------|
| 首次越域 (dz>150mm) | 52 步 | 46 步 | 42 步 |
| 域内总步数 | 56 | 77 | **85** |
| abs-mean dz | 3404mm | 840mm | **577mm** |
| dz 范围 | [-643, **+6302**] | [-649, +2318] | [-636, **+1755**] |

**出域行为质变**: 500-iter 一次越域即单调 PD 弹射逃逸到 +6.3m；1300-iter 是**有界高度振荡**（±1.75m 内往复，符号翻转 13 次）——hinge 学到的"主动保持根高"在域外依然生效，上限弹射压 3.6×。首次越域略提前 (42 vs 52) 与评测分布一致 (median=35, min=11)，无矛盾。

**踝振荡（Phase C 回退检查）**: 1300 前 25 步踝 pp 125/203/131/167 mrad (Lp/Lr/Rp/Rr)，高于 500-iter (116/177/69/73) 但**低于 ref PD 基线 (292-403 mrad)** — 无振荡回退 ✅。

**结论**: 定量指标全部通过（域内步态健康、出域高度调节泛化、无振荡回退）。视频待肉眼确认，产物在本地 job tmp (ppo1300_full.mp4 全程 / ppo1300_gait.mp4 前 150 步 / ppo1300_f*_view.png)。

### P0 物理层修复：脚部接触缺失根因与修复 (08-14)

渲染验证发现足部穿透 59cm → 诊断发现 **PhysX 脚部碰撞 box 从未接触地面**（腿链帧错位，脚 box 悬在 2-3m 高空），行走支撑来自骨盆 box + 帧错位下 RC articulation 的非物理行为。

**根因**: `createLink(parent, pose)` 的 pose 是父系相对位姿，旧代码传世界累积位姿 → 父变换每层重复应用 → 深链 link（踝）偏移 2-3m。

**五项修复**:
1. `finalize(local_poses=True)` — 传 MJCF 父系局部位姿
2. filter shader 杀掉 articulation 自碰撞（帧修正后 fallback 胶囊重叠爆炸）
3. 薄 box contactOffset 0.02→0.005（PCM 膨胀几何 10m/s 弹射）
4. 移除 ankle_pitch fallback 胶囊（插地 5cm 弹射源）
5. 训练配置 root_z_offset=+0.03（补偿动捕悬空，box 贴地起步）

**修复后**: 踝 FK 0.052m ✅；脚 box 接触地面 ✅；静态站立 60+ 步 ✅；**ref PD 基线 9.2→80.3 步**（含 145/485 步完整动捕集）。旧 hinge 策略对新物理无效（塌缩），全部训练需重新标定。

**SDK**: 原版 PhysX 5.6.1（ovphysx-0.5.9 tag，本地 /Users/kevin/code/PhysX）已恢复并重建；5.6.1 与旧 .so 物理逐位一致（指纹验证通过）。构建配方见 memory physx-foot-contact-p0-bug。

### Phase 1 完成: root_z_offset 扫描 (08-14)

修复物理后 PD 基线双峰（早死=初始穿透弹射 vs 完整动捕）。扫描结果（10 ep/值，ref PD，阈值 0.35）：

| offset | 早死(<5步) | mean length |
|--------|-----------|-------------|
| 0.02 | 5/10 | 75.6 |
| 0.03 | 1/10 | 80.3 |
| **0.04** | **0/10** ✅ | **80.7** |
| 0.05 | 0/10 | 55.1（box 离地过高，接触稀疏，高度终止提前）|

**锁定 `root_z_offset=0.04`**。PD 基线新值 **80.7 步**（旧物理 9.2）。双峰中 144-487 步的集是完整动捕跟踪。

### 修复物理重新标定计划（四阶段）

- **Phase 1** ✅ offset 扫描 → 0.04
- **Phase 2**: BC warmup 重跑（旧 bc_walking.pt 对修复物理无效）
- **Phase 3**: PPO 小规模验证 + 超参重新标定（骨架沿用宽松训练/严格评测/IGNORE_TERM=1/WORKERS=128/trust 渐升；noise/BC_COEF/hinge 需重新验证）
- **Phase 4**: 4096 production + 0.35/0.30/0.20 评测矩阵重建 + 渲染验证

旧训练结论（v6 配方数值、hinge w=2.0、37.92 等）全部基于坏物理，仅作参考，需重新验证。

### Phase 2 启动: 修复物理 BC warmup (08-14 05:27 UTC)

**Smoke 验证通过** (32 envs × 2 DDP × 100 iter, BC_ONLY + trust=0): mean length **55.4**（旧物理 ~9.2，6×）、rewards +338、0 NaN、BC loss 正常下降。iter 100 的 checkpoint 保存短写失败（容器 6.2G < 7G 阈值，已知 sda2 问题）。

**完整 BC warmup 已启动**: 4096 envs × 16 DDP × 300 iter，全新从 SONIC release 起步，trust=0 纯 ref rollout，ori/ank=0.35 + IGNORE_TERM=1 + ROOT_Z_OFFSET=0.04，checkpoint 写 /dev/shm/physx_runs/bc_walking_fixed（避开 overlay 压力）。log: /tmp/physx_bc_full_20260814_052709.log，PID 165024，ETA ~70min。

旧物理 run 目录（logs_rl 14G）暂保留未删（备份未全部确认），不阻塞新训练。

### Phase 3 关键修复: BC checkpoint 加载键错误 (08-14)

**症状**: 修复物理上 PPO/BC@trust>0 的 rollout 全员 1-2 步死，而同策略评测 (act_inference) 存活 19.6-30.8。排查链: σ 采样 → 噪声 → NPU 前向 → is_training 路径 → 注意力掩码，全部排除。

**根因**: `train_agent_trl.py` 用 `_sd.get("policy", _sd)` 加载 BC checkpoint，但 checkpoint 的 actor 存于 `policy_state_dict` 键（评测脚本读的正是这个）→ 回退加载整个 dict → 键全不匹配 (missing=55) → **BC 权重从未进入训练模型**，PPO 从未训练的 SONIC release 权重起步（输出 std 0.027 vs 评测 0.286）。

**修复**: 依次尝试 `actor_model_state_dict` / `policy_state_dict` / `policy`。加载后 missing=0。BC@trust=0.5 阶梯 length 2.05 → **14.4 @ iter 1**。

**教训**: checkpoint 加载必须打印 missing/unexpected 并核对——55 missing 从一开始就在日志里，但旧物理对近零动作有容忍度，掩盖了问题。

### 管线对比审计 + 修复 (08-14)

对比我们 fork 的管线与上游原生 SONIC（排除 env 层），8 文件有改动，修复 4 项：

1. **rollout 掩码**（ppo_trainer.py）：cur_dones=None → 从 storage 的 _orig_done 构建 episode 注意力掩码。ignore_terminations 下 env 的 masked dones 恒 False，原先会向 transformer 隐藏 episode 边界（跨集 obs 历史污染 rollout 动作）——与 BC checkpoint bug 正交的独立缺陷
2. **BC checkpoint fail-fast**：missing/unexpected 非零即报错（step-1 死亡的静默部分加载模式从此不可能再发生）
3. **配置泄漏收敛**：init_noise_std 0.15 / std_clamp 0.05-1.0（MuJoCo 适配）从全局 yaml 移回 MuJoCo 分支，恢复上游默认（0.05 / 0.001 / 0.5）
4. **删除 v14/v14b 实验块**：g1_dyn_reinit（硬编码层索引）、g1_dyn_freeze_backbone

审计结论#1（BC loss 来源路径）：aux 路径与 raw 路径的 action_mean 是同一个张量（分叉只影响 aux loss 计算，不影响动作），此前 std 0.027 vs 0.286 的观测差异实为权重未加载所致，无需代码修改。

### PPO v2 启动 (08-14 10:14 UTC)

修复物理 + 管线修复后的首个完整 PPO：从 BC@t05-200 checkpoint（t05 训练最终 length 42.2）起步，trust=1.0，v6 配方（BC_COEF=1.0 + noise 0.03/0.05 + ori/ank=0.35 + IGNORE_TERM=1），无 hinge。fail-fast 通过（missing=0）。iter 1 length=7.5（全权控制裸生存，混合值 42 的 t05 起点符合"平庸策略 PPO 信号强"的 v6 经验）。checkpoint → /dev/shm/physx_runs/ppo_v2_fixed，300 iter。

### PPO v2 重启 (08-14 14:16 UTC)

首次 PPO v2 在 iter 276 被 SIGKILL——根因 /dev/shm 99% 满（旧物理 ppo_500iter 目录 12G 未清），save 从 iter 100 起连续失败 9 次。1300 checkpoint 已备份宿主 /root/backup_model_step_001300_oldphysics.pt，旧目录删除后 shm 28% 使用率。重启全新 PPO v2（missing=0 fail-fast 通过）。首次运行曲线：7.5 → 12.7 @276 iter（爬升慢但与旧物理 v1 同节奏，v1 到 1248 iter 才 26.6）。
