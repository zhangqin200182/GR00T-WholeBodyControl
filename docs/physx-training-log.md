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

### PPO v2 第三次启动 (08-14 14:27 UTC)

第二次启动失败：HCCL 端口冲突（error code 7，16666 已被绑定）——上次 SIGKILL 残留 8 个进程占着端口。pkill -9 清干净后重启成功（iter 1 length 7.6，与首跑一致）。教训：SIGKILL 后必须确认进程清零再重启。

### PPO v2 完成 + 三阈值评测 (08-15)

300 iter 全程零 save 失败（shm 清理生效）。训练内 length 7.6→13.0（末段平顶），rewards 30.8。

**三阈值交叉评测（model_step_000300, 10 eps, 确定性）**：

| 阈值 | mean_len | median | min | max | 对照 |
|------|----------|--------|-----|-----|------|
| 0.35 | **38.50** | 39 | 17 | 64 | PD 31.0 / BC 19.6 |
| 0.30 | **36.20** | 36.5 | 15 | 62 | — |
| 0.20 | **27.60** | 22 | 11 | 57 | — |

**修复物理上的首个正向里程碑**：PPO 超越 PD 基线 24%、接近翻倍 BC。0.20 严格阈值 27.6（旧物理 hinge 1300 的 37.92 基于坏物理，不可比）。训练内/评测差距 = 探索噪声税（σ=0.05）。曲线末段平顶 → 下一杠杆：延长训练（300→1000+）或降噪声。

### 0.20 死亡模式诊断 (08-16)

v2-300 策略 @0.20 严格阈值（10 eps）：**踝水平位置 7/10、姿态 3/10、高度 0/10**。踝位置误差 0.203-0.215 vs 阈值 0.200——仅边缘超出 1.5-7.5%。旧物理的 75% 高度下沉主导死因在修复物理上**完全消失**（真实脚接触撑住根高）。结论：hinge 高度奖励大概率不需要；新瓶颈是踝跟踪精度，边际误差可能随训练延长收敛（resume 300→1000 进行中）。若 1000 iter 后仍边缘死亡，下一杠杆是踝位置塑形。

### v2-300 渲染验证 ✅ (08-16)

用户肉眼确认：**修复物理后的渲染质量大幅提升**（对比旧物理渲染的足部穿透/骨盆滑行）。v2-300 策略步态自然，前 15 步 dz±15mm 紧密跟踪。渲染产物：本地 /tmp/ppo_v2_gait.mp4 + ppo_v2_full.mp4。这是 P0 物理修复链条（帧修复→脚接触→BC→PPO→渲染）的完整闭环验证。

### 踝位置 Hinge 重训启动 (08-16 12:14 UTC)

**背景**: 1000-iter 评测：0.35→40.60 (+2.1)、0.30→37.30 (+1.1)、**0.20→23.60 (-4.0 倒退)**。死亡诊断：踝位置 9/10（0.201-0.241 vs 0.200 边际超限），延长训练不解决（宽松训练不施压踝精度）。

**实现**: `ankle_hinge_weight`（config key + `SONIC_PHYSX_ANKLE_HINGE` env var），奖励 `+w×max(0, 1-踝水平位置误差/训练阈值0.35)`——恒梯度（w/0.35=5.7/m @ w=2），镜像 height hinge 结构。冒烟测试通过（扰动踝关节 delta=0.69 ≈ 理论 0.69）。

**launch**: resume 1000→1300，v2 配方 + ANKLE_HINGE=2.0。log /tmp/physx_ppo_v2ah_20260816_121452.log。预期：critic 需数 iter 适应新标度（reward +~1.5/步），1300 后 0.20 评测对照 23.60 基线。

### 踝 Hinge 重训完成 + 评测大突破 (08-16 14:00 UTC)

1000→1300 iter，v2 配方 + SONIC_PHYSX_ANKLE_HINGE=2.0，全程无 NaN，保存零失败。训练内 length 17.9→23.5。

**三阈值交叉评测（model_step_001300.pt, 10 eps）**：

| 阈值 | 1000-iter 基线 | 踝 Hinge 1300 | Δ |
|------|---------------|---------------|-----|
| 0.35 | 40.60 | **62.30** | **+21.7 (+53%)** |
| 0.30 | 37.30 | **58.20** | **+20.9 (+56%)** |
| 0.20 | 23.60 | **32.90** | **+9.3 (+39%)** |

**死亡诊断**：踝位置仍 9/10（0.201-0.231 边缘穿越），但长度翻倍（20-66 vs 11-32）——hinge 把踝跟踪精度推到了阈值线附近，边际穿越成为长尾。0.20 的 32.9 已超越 300-iter 最好成绩（27.6），且修复物理上的历史新高。

### 踝速度惩罚重训启动 (08-17 01:34 UTC)

**背景**: 踝 hinge 1300 的渲染暴露 Phase C 振荡回退——用户观察"很抖"，定量确认：每步踝抖动 32-116 mrad（300-iter 仅 19-28，ref PD 基线 pp 292-403），踝 pp 达 523-1017 mrad。机制：位置 hinge 恒梯度教会策略用高频泵动贴住踝位置（位置奖励不惩罚振荡）。

**实现**: `ankle_vel_penalty_weight`（+`SONIC_PHYSX_ANKLE_VEL_PENALTY`），`-w×mean(踝 qvel²)`，w=0.05（正常运动 mean_sq 3.0 vs hinge 抖动 4.4）。resume 1300→1600，HINGE=2.0 + VEL_PENALTY=0.05。

**预期**: 抖动降回 ~20 mrad/步，0.35/0.30/0.20 评测保持或微降（抖动捷径被封，策略需学平滑的踝控制）。

### 踝速度惩罚重训启动 (08-17 01:34 UTC)

**背景**: 踝 hinge 1300 的渲染暴露 Phase C 振荡回退——用户观察"很抖"，定量确认：每步踝抖动 32-116 mrad（300-iter 仅 19-28，ref PD 基线 pp 292-403），踝 pp 达 523-1017 mrad。机制：位置 hinge 恒梯度教会策略用高频泵动贴住踝位置（位置奖励不惩罚振荡）。

**实现**: `ankle_vel_penalty_weight`（+`SONIC_PHYSX_ANKLE_VEL_PENALTY`），`-w×mean(踝 qvel²)`，w=0.05（正常运动 mean_sq 3.0 vs hinge 抖动 4.4）。resume 1300→1600，HINGE=2.0 + VEL_PENALTY=0.05。

**预期**: 抖动降回 ~20 mrad/步，0.35/0.30/0.20 评测保持或微降（抖动捷径被封，策略需学平滑的踝控制）。

### 动作空间审计 + 计划调整 (08-17)

采纳 `docs/physx-action-space-audit-and-plan.md`（同事实验审计 + 我们动作空间诊断）。核心发现：

1. **动作空间错配**：我们 `jm+jh·a` vs Isaac `default+0.25·e/k·a`，逐关节 0.6×~21.7× 失真——官方权重的技能语义被扭曲，所有训练的起点先验是坏的
2. **kp×15000 是坏物理时代（脚修复前）扫出的暴力补偿，修复后从未重标定**（PD 9.2→80.7 暗示正确增益可能小得多）
3. 力矩限幅统一 50 vs Isaac 逐组 139/88/50/25/5
4. 同事 E3 实锤：同权重同引擎只换动作空间 → 4-7 步摔变零摔倒

**计划调整**：
- 旧空间训练降级为收尾基线（vel-penalty 1600 评测照做，数值作对照）
- Phase A：三个单变量零训练实验（①只换动作映射 ②kp 重标定 ③逐关节限幅），官方权重零样本 + ref PD 重测
- Phase B：决策树选主线 → Phase C 对齐空间重标定 → Phase D：Isaac 回传 + 跨片段评测（防过拟合必修课）
- 保留：物理修复、管线修复、方法论；作废风险：旧 checkpoint 数值、hinge 具体数值（方法论保留）

### 更正 (08-17)：所有 PhysX 训练实际从随机权重起步

此前日志中"全新从 SONIC release 起步"的记录（BC warmup、PPO 启动条目）**全部更正**：代码存在 release 加载路径（`pretrained_model` config），但所有 PhysX 运行从未配置它——BC-300 的 encoder 权重与随机初始化逐张量相同（30/30），所有训练（含 62.3/58.2/32.9 及本文档后续的 209.4/165.0/42.1）均为随机初始化的 37M 模型从零学出。详见 `docs/physx-alignment-experiment-plan.md`。

### 旧空间最终基线 (08-17)

vel-penalty 1600 完成（训练内 27.1）。**三阈值评测：209.4/165.0/42.1 @ 0.35/0.30/0.20**——速度惩罚后宽松阈值三倍增长（62.3→209.4），0.20 从 32.9→42.1。此为旧动作空间的最终对照基线，进入对齐实验阶段（Phase A）。

### 对齐实验进展 (08-17)

**前置完成**：release 权重加载（stub loader，missing=0）、部署栈参数表提取（与 CFG/E3 三方一致，3 findings）、obs 对比进行中。

**实验一**（动作空间对齐，零训练）：release 零样本 1.00 → **1.90** 步——单独换空间不够。
**实验二**（解析驱动增益 kp=k_isaac/M_eff）：1.90 → **13.8**（mult 1.0）→ **20.2**（mult 10）→ **25.6**（kd×8）。release 从一步死到逼近 PD 基线（27.0）。kd 侧阻尼不足是关键（kd_mult 1→8 提升 25%）。
**实验三**（XML hip 限幅 88→139）：23.3，噪声级。

**当前最优配置**：isaac_action_space + DRIVE_ANALYTICAL=1 + MULT=10 + KD_MULT=8，release 零样本 ~25 步（"数十步 = 部分确认"区）。下一步：obs 对比报告（若发现错位则修）+ 增益精细扫描 + 若 release 能到数百步则环境冻结。

### obs 对齐修复 + 负结果 (08-17)

obs 审计代理报告 6 处与上游 Isaac 定义的错位，已修复 4 处（`physx_env.py`）：

1. **actor obs 块顺序**：`[gdh, avh, jph, jvh, ah]` → `[avh, jph, jvh, ah, gdh]`（gravity 最后）。依据上游 `observations.py` PolicyAtmCfg 注释："Order matches PolicyCfg: base_ang_vel, joint_pos, joint_vel, actions, gravity_dir"。
2. **joint_pos 相对化**：`jpos` → `jpos - _act_offset`（q − q_default，`joint_pos_wo_hand = joint_pos - default_joint_pos` 上游同款）。
3. **cmd_mf / cmd_nonflat 关节序**：mujoco 序 → isaaclab 序（`ISAAC_REORDER` 置换，580 维 critic/tokenizer 块）；`cmd_lower` 同步改 12 腿关节 isaaclab 序。
4. **critic base_lin_vel**：世界系差分 → body 系（`quat_apply(quat_inv(root_quat), ...)`）。

未修 2 处（低影响）：SMPL 块全零（原始数据缺失）、obs 噪声注入缺失。

**评测（负结果）**：release 零样本同配置（isaac space + ANALYTICAL + MULT=10 + KD_MULT=8，12 episodes）**25.08 步 vs 修复前 25.6** —— 无变化（噪声级）。结论：**obs 错位不是 release 零样本的瓶颈**。release 策略在 ~25 步 ≈ PD 基线 27.0，行为仍是"跟踪 ref 姿势"的兜底，未展现其学到的技能 → 剩余差距在**物理（驱动动力学）层**，不在 obs 层。mujoco_env.py 同期基线（209/165/42）同样是 gravity-first 布局（随机权重训练对内部一致性不敏感，不影响其数值）。

### 增益精细扫描完成 + 平台期确认 (08-17)

7 配置全扫（isaac space + ANALYTICAL，obs 修复后环境，各 10 episodes）：

| mult | kd_mult | mean_len | mean_rew |
|------|---------|----------|----------|
| 10 | 8（对照） | 24.50 | 95.3 |
| 10 | 16 | 23.70 | 109.2 |
| 10 | 32 | 24.20 | 125.5 |
| 10 | 64 | 25.10 | 138.1 |
| 5 | 16 | 23.80 | 109.0 |
| 20 | 16 | 24.00 | 111.3 |
| 30 | 16 | 24.00 | 110.6 |

**结论**：存活步数全网格饱和在 **24-25 步**（噪声带 23.7-25.1）；kd↑ 只提升平滑度（reward 95→138 单调升）不提升存活。**增益杠杆已耗尽**——release 零样本 ~25 步 ≈ PD 基线 27 是**物理保真度差距**（驱动动力学近似/接触/惯量），不是增益或 obs 问题。实验二完成，进入 #18 决策。

### Isaac 动力学对齐启动 (08-17)

用户定调：与 Isaac 动力学模型逐项对齐（同一引擎、两组配置）。路径：①测 FORCE 驱动 → ②静态参数审计（每改一项 release 零样本重测，25 步为回归基线）。

**① FORCE 驱动模式测试（负结果）**：loader 增加 FORCE 分支（kp/kd 原始 Isaac 值 + effort 限幅 = Isaac 驱动语义原样），bindings 早已支持 eFORCE。12 episodes：
- FORCE release 零样本 **17.33**（vs analytical 25.6）——更差
- FORCE ref PD **27.17**（≈ analytical PD 27.0）
- eACCELERATION 原始 kp/kd（RAW，"惯量化"假设）release **14.50**、PD 15.92——太软，直接垮
- **分析**：policy 在 FORCE-raw 下比纯 PD 还差 → 力矩域软 PD 不是 policy 的世界；analytical eACCELERATION（mult=10 有效 kp≈10×Isaac）仍是当前最优 → 剩余差距不在驱动增益单变量。

**② 静态参数审计（XML vs Isaac main.urdf + g1.py CFG）**：找到 5 类差距：
1. **关节摩擦**：XML 29 关节 frictionloss 0.2-3.0（P1.5 时代为 MuJoCo 稳定性加的），Isaac main.urdf **全 0**。已改 XML → 0。
2. **torso 质量**：XML 9.598 vs Isaac 6.78 + head 1.036 = **7.816**（XML torso 无 head link，质量含 head + 1.78kg 幻影）。已改 mass → 7.816（diaginertia 保留，已含 head 的 Steiner 项）。waist_roll 0.047→0.086 同步。
3. **求解器**：Isaac pos_iters=8 **vel_iters=4**；我们 8/1。待改（batch 2）。
4. **dt**：Isaac sim_dt=0.005 decimation=4（50Hz）；我们 0.001961×10（51Hz）。待改（batch 2，需重扫增益）。
5. **armature**：Isaac 逐关节 armature（0.0251/0.0102/0.0036×2/0.00425），XML/bindings 无。待改（batch 3，需 bindings 支持）。
另有 velocity_limit_sim 逐组（20-37 rad/s）、max_depenetration_velocity=1.0、wrist_yaw 手部质量模型差异（0.255 vs Isaac 0.085+手链 ~0.78）——记入 audit 清单。

### Isaac 动力学对齐进展汇总 (08-17 晚)

**全杠杆评测表**（release 零样本，isaac space，12 episodes，ori/ank=0.35）：

| 配置 | release | ref PD | 备注 |
|------|---------|--------|------|
| analytical mult=10 kd=8（基线） | 25.6 | 27.0 | 实验二最优 |
| + obs 对齐修复 | 25.08 | — | 无变化 |
| FORCE 驱动（原始 kp/kd+effort） | 17.33 | 27.17 | 更差，力矩域软 PD 非 policy 世界 |
| eACCELERATION 原始 kp/kd | 14.50 | 15.92 | 太软，直接垮 |
| + XML 摩擦=0 + torso 7.816 | **25.58** | 28.00 | PD 略升（reward 142→157），policy 无变化 |
| + vel_iters=4 | 24.92 | — | 无变化 |
| + Isaac dt 0.005×4 | 22.83 | — | 略降（增益需按新 dt 重扫） |

**当前状态**：release 零样本硬钉在 ~25 步 ≈ PD 基线。已排除：动作空间（1.0→1.9 是主因，已修）、obs 布局、增益全网格、FORCE/RAW 驱动语义、关节摩擦、torso 质量、求解器迭代数、dt。**剩余未测**：armature（bindings 需加 setArmature）、velocity limits（bindings 无 maxJointVelocity）、depenetration velocity、contactOffset（我们 0.005，Isaac 默认待查）。

**下一步**：① release 零样本死亡模式诊断（25 步时挂在哪——ori/ank_pos/高度，与训练期 0.20 诊断的"踝位置漂移 70%"对照）② batch 3：armature 支持（C++ 改动）③ 若 armature 后仍无突破 → 按用户框架结论："残余差距在引擎时序层"，进入决策。

### Batch 3：armature / velocity limits / depenetration (08-17 晚)

**实现**（commit ed2a03b）：bindings 暴露 `setArmature` / `setMaxJointVelocity`（PxArticulationJointReducedCoordinate 原生 API，默认 0.0 / 100.0）与逐 link `setMaxDepenetrationVelocity`；loader 新增 `_ISAAC_ARMATURE` / `_ISAAC_VEL_LIMIT` 逐关节表（值取自 g1.py 电机类常量，与 policy_parameters.hpp 一致；Isaac 的 kp/kd 本身由 armature 推导：kp=armature·ω²）。默认开启，`SONIC_PHYSX_ARMATURE/_VEL_LIMIT/_DEPEN=0` 可关。同时补齐死亡模式诊断 plumbing：`_check_termination` 的 term_reason 经 info 传出，cross_eval 逐 episode 打印原因并分类计数。

**评测矩阵**（release / ref PD，0.35/0.35，12 eps，analytical 10/8 + isaac space）：

| 配置 | release | PD | 备注 |
|------|---------|-----|------|
| 基线（三开关 OFF）| 25.58 (rew 97.1) | 28.00 (157.2) | 复现历史 25.6/27.0 ✓ |
| +armature | 26.67 (101.9) | **34.67** (199.1) | PD +24% 是硬信号 |
| +armature mult∈{1,5,20} | 26.00/26.75/26.00 | — | 存活对 mult 平坦；死亡构成随 mult 漂移（低刚度 ori↑ ank↓）|
| +armature+vel_limit | 26.67 | — | 行走速度下从未触发（20-37 rad/s 远超）|
| **+armature+depen** | **35.25** (123.0) | 34.67（与 arm-only 相同）| 历史新高 |
| +armature+vel+depen | 26.67（12eps 异常，见下）| — | — |
| FORCE+armature | 11.58 | 27.42 | FORCE 路线再次证伪 |

**复测（24 eps）**：armdp 32.92（seed 0）/ 29.21（seed 1）——depen 效果两 seed 确认；armall24 与 armdp24 **逐 episode 完全相同**（vel_limit 从不触发）；basedp12（depen 单独、无 armature）25.67 ≈ 基线——**depen 单独无效，armature×depen 是协同杠杆**（arm-only 26.67 / depen-only 25.67 / 组合 29.2-35.3）。armdp_pd 与 arm_pd 相同（34.67）——PD 追踪脚底干净无穿透事件，与策略侧形成机制对照。

**12-eps armall 异常记录**：矩阵 armall_rel（v=1）ep0=11 vs armdp_rel ep0=60，同配置跨进程轨迹不同（vel=1 运行存在跨进程非确定性，疑似速度限幅冲量路径）；24-eps 复测 armall == armdp 说明常态下无影响。保留默认 ON（Isaac 语义），训练中仅摔落瞬间触发，符合 Isaac velocity_limit_sim 行为。

**死亡模式诊断结论**：release 零样本 25 步死亡 ank_pos 11/12（92%），比训练期 70% 更集中——踝水平位置漂移是 release 侧第一死因；armature 修复后（mult=5）ank_pos 降至 7/12、ori 升至 5/12，两个竞争短板。

**配置决策**：armature + depen 保留为默认（已在代码中默认开启）。release 零样本 25.58 → 32.92（+29%，双 seed 确认）、PD 28.0 → 34.7（+24%）。仍未达「数百步」全确认区——静态参数审计清单（armature 为最后一个大杠杆）基本耗尽，剩余 contactOffset 待查 / wrist_yaw 手质量（总量差 5%）预期低影响。

### 同事训练配方审计 (08-17 深夜)

**背景**：复盘「同事上千轮站稳的借鉴是否覆盖」——本次对齐工作（实验一 + batch 3）只覆盖了**接口参数层**（动作空间 0.25·e/k、armature 四档、effort 限幅、velocity_limit_sim——均来自同事 E3 代码与 Isaac CFG 共用的电机常量表 `g1.py` / `policy_parameters.hpp`），**训练配方层未覆盖**。审计 `gear_sonic/envs/mujoco_env.py`（同事 MuJoCo 环境，209/165/42 的评测环境）逐项对比：

| 项 | 同事 mujoco_env.py | 我们 PhysX | 判定 |
|----|-------------------|-----------|------|
| feet_acc 权重 | **-2.5e-9**（Isaac 值 1000× 降权，mujoco_env.py:359-365）| -2.5e-6（Isaac 原值，physx_env.py:449）| **未借鉴**——同事注释：原权重下脚接触尖峰 ~1e8 使该项占 97% 惩罚，逼出「早死少扣分」退化策略 |
| alive_bonus | **4.0**（2.0→4.0 逐级上调，:81-87 记录调参历程）| 0.0（train_agent_trl.py:387,415）| **未借鉴** |
| 训练终止阈值 | ORI **0.5** / ANK **0.6** / H_MULT **6.0**（v11→v13→v15 三级放宽，:442-447）| 0.35/0.35/1.5 | **未借鉴**（同事幅度更大）|
| PD 增益 | kp=100/kd=5 均匀 + 每 substep 重算力矩（:62,207-215）| PhysX drive 内部逐 substep 解算 | 不适用（FORCE 路线已证伪；「kd 须响应中途加速」语义 PhysX 已处理）|
| solver iterations=200 | MuJoCo 专用（:73）| — | 不适用 |
| 奖励 r1-r11 权重 | 与 Isaac 一致（:323-357）| 与 Isaac 一致 | 无需借鉴 |
| r14/r15 hinge 塑形 | 无 | 有（高度/踝 hinge，config 开关）| 我们独有 |

**结论**：同事三项训练侧参数（feet_acc 降权 / alive_bonus / 阈值放宽）本质是**「扭曲环境的补偿器」**——其物理有缺陷（针尖脚），权重学不稳，靠训练侧放水让学习走得动，属「权重适应环境」路线产物。我们的路线相反：环境对齐 + 训练侧保持 Isaac 原语义，使「Phase C vs 209.4」对比纯粹度量环境对齐价值。**三项参数定位为 Phase C 平顶时的后备诊断，而非起步配置**：对齐做对了理论上不需要它们；若 Isaac 原语义下平顶、加补偿后突破，即诊断出环境对齐仍不充分（残余 gap 被训练侧放水掩盖）。

### 宽松阈值测试：release 技能全水平被压制 (08-17)

**背景**：回答"32.9≈PD 是严格阈值的边际问题，还是技能整体被压制"——release 零样本 + ref PD 在三个阈值水平各 12 eps（isaac space + analytical 10/8 + armature/depen）。

| 阈值 | release | PD | 判读 |
|------|---------|-----|------|
| 0.35/0.35/1.5 | 35.25 | 34.67 | 打平 |
| 0.5/0.5/3.0 | 47.00 | 42.75 | 差距不拉开 |
| 0.5/0.6/6.0 | 42.92 | **49.42** | **PD 反超** |

**判决**：无一次走满片段（max 94 步）；reward 全倒挂（PD 高 60-90%）；宽松阈值下死因从踝漂移转为 ori 8/12 + height 7/12（真实摔倒，6× 高度阈值都拦不住）。**技能在所有阈值水平被压制 → 环境未通过验收 → Phase C 的「对齐完成」前提证伪 → Isaac 动态数据升级为关键路径。**

### 重大发现：评测数据只有 2 条动捕 + 评测确定性失效 (08-17)

1. **容器/本地只有 2 条动捕**（`walk_forward_amateur_001` + `_M` 变体，官方 HF demo 集）。此前所有评测数字（32.9、35.25、26.67 等）都是 2-片段测量。「12 条动捕」从来不存在（记忆里的 12 PKL 计入了 macOS 垃圾文件）。
2. **cross_eval 确定性失效**：`_load_motions` 用 PID 洗牌（跨进程随机底序），eval 的 mseed permutation 不能消除底序随机性。N=2 时每次运行 = 两排列硬币翻转 → 结果双峰：排列 A=35.25 / 排列 B=26.67，分支内逐集逐位可复现。旁证：N≤4 时 `RandomState(0)`≡`RandomState(2)` 的 permutation（故 mseed 0/2 等价）。
3. **batch-3 误诊纠正**：矩阵的"vel=1 跨进程非确定性"实为排列硬币翻转，vel_limit 无罪。**depen 结论被分支污染**：arm-only 测在 B 分支（26.67）、arm+depen 测在 A 分支（35.25）——"depen 协同杠杆"（25.6→32.9）很可能是分支跳变；B 分支内 arm+depen == arm-only（零增益）。PD 侧结论（armature +24%）不受影响（PD 三次运行 34.67 逐位相同，对分支不敏感）。

### 修复：评测确定性 + 数据补充 (08-17)

1. **代码修复**（physx_env.py / mujoco_env.py / physx_cross_eval.py）：loader 洗牌改固定 seed=0；`_sample_motion` 记录 `_cur_motion_id`；cross_eval 重排时同步 `_motion_ids` 并逐集打印 `clip=` 归属。双进程验证：逐集逐位一致（33.00/33.00，IDENTICAL diff）✅
2. **数据补充**：宿主机 `/root/sonic-data/robot_filtered` 有 **89,464 条**已转换片段（5.1GB，BONES-SEED 转换产物；另有 13.9GB `g1.tar.gz` 原始包）。按排序均匀抽样 500 条步行片段拷入容器 sample_data（现 502 条，35MB）。

### 新回归基线（500 片段、确定性、mseed=0，0.35/0.35，12 eps）

| 策略 | mean_len | mean_rew | 备注 |
|------|----------|----------|------|
| release 零样本 | **31.83** | 115.4 | 12 个片段全部不同（含拐杖/伤腿/负重等病理步态）；新死因 wrist_yaw_h |
| ref PD | **37.00** | 203.5 | **PD 反超 release 16%** |

> 最终数字以 loader 加 `sorted()`（文件名确定序）后的重跑为准：31.83/37.00。此前 27.58/36.42 为 glob 文件系统序下的测量，已作废。

**旧数字（32.9/35.25/26.67 及 27.58/36.42）正式退役。** 多样化片段上训练策略不如手写跟踪器——技能压缩的强信号（2 片段时打平，500 片段时倒挂 16%）。

### Sequential 模式 + 固定 12 片段集 (08-18)

**跨引擎逐集对照**需要 episode↔片段固定配对：cross_eval 新增 `--pkl`（指定片段目录）与 `--sequential`（episode i ↔ loader 序第 i 个片段，无随机采样）；loader 序 = 文件名 sorted + RandomState(0) 洗牌（与文件系统无关，已本地预计算并实跑验证逐位一致）。env 侧 `_sample_motion` 支持 `_forced_idx` 覆盖随机采样。

**固定 12 片段结果**（/sample_data/robot_filtered_fixed12，0.35/0.35）：

| 策略 | mean_len | mean_rew |
|------|----------|----------|
| release | 25.67 | 99.3 |
| PD | **34.33** | 177.2 |

PD 在 12/12 片段中 11 个反超；差距最大的是病理步态（拐杖 50 vs 21、伤腿 57 vs 29）。逐集表见 `docs/physx-isaac-data-collection.md` 附录 B。

### Isaac 基准数据收集请求 (08-18)

撰写 `docs/physx-isaac-data-collection.md`：完整的数据收集协议（P0 基准 rollout 字段规范 / P1 单关节驱动响应）、导出钩子参考代码、回传 checklist、12 片段确定序配对表。数据用途四条：①基准裁决（release-in-Isaac 步数 vs 31.83）②轨迹级 diff 定位发散关节/相位 ③状态注入力矩对比 ④驱动响应反解。**这是当前定位残余 gap 的关键路径，等同事执行。**

### Isaac 数据回传 + 基准裁决① (08-18 晚)

**数据包**：24 集 P0（release+pd × 12 片段，t=0、终止中和、字段齐全）+ P1 27 组驱动响应（3 关节 × 阶跃 ±0.05/0.1/0.2 + 正弦 3 频，root 固定漂移 ≤2.8mm）+ protocol_notes/PD 说明/version/commands/obs_step0。质量极高（PD identity 验证、烟测接触力 343N≈体重、P1 质检 27/27）。

**三个采集侧发现（后处理解决，数据有效）**：
1. **航向重置约定**：Isaac 机器人从 USD 资产默认朝向（-85° yaw）起步、从不与 ref 初始航向对齐（策略学的是航向相对跟踪——对齐起步的 270° 转身片段全程 ori 误差 2-6°，错位起步片段 60 步内不回头修正）。我们的 env reset 强制 root=ref 朝向。→ 后验判定先做**航向相对化**（ref 轨迹旋转到机器人初始航向系）。
2. **上臂关节约定偏移**：Isaac 资产腕/肘/肩 yaw 零位与 mocap 约定差 ~10-65°（恒定，策略从不跟踪 mocap 腕值）。腿/踝链干净（3-16°）。→ wrist-h 检查对 Isaac 数据剔除，踝/高/姿态检查保留。
3. **出生瞬态**：Isaac 从资产默认位姿起步（非 ref[0]），个别片段第 1 步踝误差 0.4m，策略 ~10 步收敛到 0.1（injured_torso 实测 0.401→0.103@step20）。→ grace 敏感性分析（0/5/10 步）。

**后验脚本自洽验证**：`scripts/adjudicate_isaac_baseline.py` 在我们自己的 save npz 上复现 live 评测**逐位一致**（25.67 == 25.67）✅

**裁决结果**（航向对齐 + 腕检查剔除 + grace 敏感性）：

| grace | Isaac release | Isaac PD | release/PD | vs 我们 release 25.67 |
|-------|--------------|----------|-----------|----------------------|
| 0 | **87.33** | 18.58 | **4.70×** | **3.40×** |
| 5 | 90.75 | 20.33 | 4.46× | 3.54× |
| 10 | 94.67 | 20.67 | 4.58× | 3.69× |

**判决：技能压缩叙事证实并定量化，不反转。**
1. Isaac 侧 release 以 **4.5-4.7× 碾压 PD**——技能真实且强。
2. 我们环境把 release 压到 25.67，**gap = 3.4-3.7×**。
3. **我们的 PD 基线被美化**：34.33 vs Isaac 18.58——×15000 暴力增益的残留，朴素跟踪器在我们环境活得比 Isaac 长 1.8×。真实对比下我们侧 release/PD 比值被双向扭曲（技能被压 + 基线被抬）。
4. 死因结构两侧一致（踝位置漂移主导、边际超限 0.35-0.41），Isaac 侧同样失败模式发生晚 ~3.5×。

**后续**：②轨迹级 diff（同片段逐帧 qpos 对比定位发散关节）→ ③a/③b 力矩对比（P1 已到货，根固定响应可直接反解 ④）。

### 同事复查：采集 bug 实证 + ④ 增益反解 + 实测增益实验（负结果）(08-18 深夜)

**采集 bug 实证确认**：按批次分组（4 批 × 3 env），批内三 env 的 qpos 轨迹差异模式跨批**逐位重复**（max|diff|=0.771743 / 1.047849）——策略每批消费同样的 3 条片段，COLLECT_CLIPS 只改了记录侧 ref。**附录 B 的 Isaac release 逐集列作废**（PD 列与 ③a/③b 不受影响）。已请求同事：①修 obs 侧注入（策略消费的 motion sampling 同受 COLLECT_CLIPS 控制）+ 记录每 env 实际采到的动捕 index，重采 release P0；②dump 全部 29 关节运行时 stiffness/damping。

**④ 增益反解**（同事 P1 拟合，R²=0.91-0.999）：Isaac 运行时真实增益与 CFG 公式严重不符——踝 kp 实测 **765**（CFG 28.5，公式差 27×）、膝 445（CFG 99.1）、髋 kp 143.3。我们当前踝 kp 285 比 Isaac **软 2.7×**（与"踝位置 70% 死因"呼应），膝 991 硬 2.2×。

**实测增益实验（负结果）**：`SONIC_PHYSX_PD_MEASURED=1`（3 关节+镜像写实测值，MULT=1/KD_MULT=1）固定 12 复测：

| 配置 | release | PD |
|------|---------|-----|
| 基线 mult=10/kd=8 | 25.67 | 34.33 |
| 实测增益 | **19.08（-26%）** | **21.83（-36%）** |

**连 PD 都掉 36%**——Isaac 真实增益在我们植物上是次优的。**结论**：增益错配不是技能压缩的正杠杆；我们的植物动力学与 Isaac 的差异大到"Isaac 最优增益不可移植"。释放差距的定位回归**策略输入层（obs）**——同事主论题成立。下一步等同事：obs 侧修复重采（策略输入=记录片段，逐 env 动捕 index）+ 29 关节增益 dump；我们侧可做：我们自己的 obs_step0 导出与 Isaac 侧逐集对比（同片段同 t=0 的 930 维 obs 直接 diff，定位 obs 层错位的具体分量）。

### obs_step0 取证（部分完成）(08-19)

`scripts/obs_step0_forensics.py`：cross_eval 新增 obs_step0 导出；取证方法 = 用 obs 的 ah 帧指纹在记录轨迹的 action_raw 序列中定位捕获时刻，再用同 10 帧状态按我们公式重建 obs 逐块 diff。

**结果**：
1. ✅ **布局层匹配**：Isaac 的 930 维 obs 解析为块优先 [avh,jph,jvh,ah,gdh]×10 帧（gravity 末位、gdh≈[0,0,−1]）——obs 审计的块顺序修复正确。
2. ✅ 我们侧构造自洽验证：jph = ref[0]−default 逐位吻合（2.4e-8）、零历史帧模式正确。
3. ⛔ **内容层对比被采集 bug 阻断**：Isaac 的 3 个 obs 行来自记录开始前的未知时刻（ah 指纹与全部 12 条记录轨迹的 action_raw 都不匹配，err 0.78-1.44）——obs 行的捕获时刻未定义。

**给同事的重采要求追加**：obs_step0 在**定义明确的时刻**捕获（reset 后第一步，逐集导出）+ 记录消费的动捕 index。我们侧工具链已就绪，数据一到即可逐块 diff。

### ③b 驱动语义判别：三配置 P1 复现 + 增益层完整排除 (08-19)

我们侧 P1 复现（root 固定+锁关节+同 27 组激励，`physx_p1_replica.py`；新增 `wake_up` binding——锁态求解器睡眠会清零驱动力）vs Isaac P1：

| 0.1rad 阶跃 | Isaac | A(标准10/8) | B(ANA+拟合) | C(FORCE+拟合) |
|---|---|---|---|---|
| hip 上升/超调 | 45ms/10.5% | 1250ms/0% | 145ms/63% | 90ms/1.6% |
| knee | 35ms/1.0% | 1245ms/0% | 155ms/0% | 155ms/0% |
| ankle | 35ms/0% | 1360ms/0% | 185ms/0% | 160ms/0% |
| 5Hz 幅值比 hip/knee/ankle | 1.00/0.94/0.91 | 0.06/0.06/0.06 | 0.17/0.48/0.45 | 0.70/0.48/0.47 |

**发现**：①A 的真实有效 kd = kd_mult×mult×kd = 504（膝），比 Isaac 24.7 高 20×——同事表漏了 ×mult 因子；②隔离尺度下 C 最接近（FORCE 语义受支持）；③B/C 仍比 Isaac 慢 2-5× → 植物有效惯量/驱动模型残留差异，非纯增益。

**FORCE+拟合 release 复测：18.83（-27%，ori 死亡主导）**。增益层完整排除（A/B/C 三配置全负）；隔离尺度（C 最优）与整机尺度（C 最差）结论相反 → 差距在闭环整机行为，不在增益值。附带：cache eFORCE 读回=外部力通道（我们导出的 applied_torque 实为全零），真实力矩读回需 eLINK_INCOMING_JOINT_FORCE 投影（C++ 排队）。

### Re-matrix：armature/depen 效应重测 (08-17 晚，确定性 + 500 片段)

batch-3 被分支污染的结论在新底座上重测（同 mseed=0、同 12 片段、逐集 clip 归属可溯源）：

| 配置 | release | PD | 效应判定 |
|------|---------|-----|---------|
| 全关（base）| 24.33 | 28.83 | — |
| 只 armature | **27.58** | — | **+3.25（+13%）release 侧唯一有效杠杆** |
| 只 depen | 24.33 | — | 0（与 base 逐位相同）|
| arm+depen | **27.58** | **36.42** | depen 叠加 0；PD 侧 armature +26%（+7.59 步）|

**结论**：batch-3 的「depen 协同杠杆」正式证伪——depen 在任何组合下零增益，此前 25.6→32.9 的"提升"是排列硬币翻转假象。**唯一有效的静态参数是 armature**（release +13%、PD +26%）。config 决策：depen 保持默认开（无副作用）或关闭均可，不再作为"协同杠杆"引用。


### D1 实验：Isaac 真驱动配置（FORCE + CFG + armature ON）(08-20)

**① P1 ③b 复现完成**（`physx_p1_replica_manual.py`：manual 滞后显式 PD + armature ON + root 1.05，DT=5ms，gravity off）：

1. **驱动定律逐位复现**：τ[k] = kp·e[k−1] − kd·v[k−1]，17 组全部 4 位小数一致（tau0 readback 2.8500 vs 2.8501 等）。
2. **🔴 Isaac 质量阵含 armature**（推翻"Isaac 物理无 armature"结论）：显式定律解读下踝 M_eff 0.0111 vs armature-ON 模型 0.0100（10% 内）、髋 0.065 vs arm-ON 自由模型 0.078（17%）。此前"无 armature"读数是隐式/显式解读伪影（首子步 v0=0 使 h·kd 项消失，第一子步无法判别驱动语义）。
3. **离散稳定性解释 ARMATURE=0 爆炸**：滞后显式 PD 在 200Hz 的极点分析——自由踝 M=0.0023 → |λ|=3.2（发散）；armature-ON（M=0.0095-0.081）全部 |λ|<1（稳定）。D1 原计划 ARMATURE=0 作废，armature 保留。
4. **我们最终配置完全稳定**：peak=tau0 精确（零发散）、± 对称、线性；首子步 M_eff 与 arm-ON 自由模型吻合 1-3.5%（髋 0.0807/0.078、膝 0.0644/0.0645、踝 0.00962/0.00954）。
5. **🔴 Isaac 记录通道互不自洽（重大采集侧发现）**：dq/h = 3.4-4.3× v（从第一子步起，踝恒定 3.37）；dv ≠ h·τ/M（5Hz 检查误差 0.05 > 信号 0.033）；q 二阶差与 τ 反号（等效负质量）。**记录的是驱动内部状态而非物理轨迹**——qpos 通道被放大约 3.4-4×，真实物理响应无法从通道恢复。
6. **真实 rise90 重构**：从 v 通道 cumsum 推算，45ms 时真实位移仅 ~22% 步进 → 真实 rise90 ≈ 155ms ≈ 我们的 135-155ms。他们表观 35-45ms（此前所有"我们慢 2-5×"结论的依据）是通道伪影。
7. 5Hz 正弦：我们 amp 0.47/相位 −59°（与滞后环解析解 0.44/−68° 一致 ✓）；Isaac 通道修正后 ~0.21-0.27/−31°（~2× 差距），但通道已证明被破坏，修正值不可靠。超调（髋 10.5%、膝 1%、踝 0%）、ss_err 全部匹配。

**判决**：驱动定律 + 增益 + armature + 阶跃响应（行走主导频段）全部对齐；③b 表观差距（45 vs 155ms）为 Isaac 通道伪影，**引擎侧无慢速差距**。D1 复测配置合法：FORCE + CFG 原始增益 + armature ON + vel_iters=4。

**②③ release/PD 复测**（vs 25.67 / 34.33）运行中，结果待补。

### D1 ②③ 复测结果 (08-20)

| 配置 | release | PD | vs 基线 |
|------|---------|-----|---------|
| 基线 ANALYTICAL mult=10/kd=8 | 25.67 | 34.33 | — |
| **D1: FORCE+CFG+armature ON+vel4** | **15.67** | **28.75** | **−39% / −16%** |
| (Isaac 侧同口径) | 87.33 | 18.58 | — |

- D1 release 死因：ank_pos=5, body_h=4, ori=3, height=1（比基线更分散、更早摔）。
- **PD 28.75 落在 Isaac 18.58 与基线 34.33 之间**——忠实驱动下我们的 PD 向 Isaac 绝对水平靠拢（gap 15.75→10.17），但 release 在忠实驱动下被压得更狠（15.67 < PD）。
- **判读**：FORCE+CFG 软驱动下策略力矩按植物真实惯量生效——我们的植物髋/膝 M_eff 仍比 Isaac 重 17%/32%（首子步测量）→ 跟踪变弱、更早摔。基线 mult=10 的刚度掩盖了这个惯量差。**残留差距定位植物侧，与 torso 静态 bug（+1.036kg、CoM z +29mm、俯仰惯量 2.1-2.4×）方向一致**——髋驱动腿对抗的躯干俯仰惯量被放大。
- **下一步**：torso/waist/wrist 静态修复（任务 #7）→ P1 髋 M_eff 复验（预测 0.0807→≈0.065 与 Isaac 汇合）→ release/PD 复测。D1 实验价值：驱动定律/armature/通道伪影三大结论 + 排除驱动层，剩余唯一已知物理差异 = torso 静态 bug。

### D2 结果：torso/waist/wrist 静态修复 (08-19) —— ✅ 修复验证通过，行为 null

**修复内容**（任务 #7 收尾）：手部质量错位归并的精确修正——torso 7.816→按 URDF 归并（head 1.036 与 torso 精确 lumped，CoM z 0.1799→0.1746 XML 系）、waist_yaw 0.244→0.214、wrist 0.255→0.781（wrist_yaw+wrist_roll+wrist_pitch+hand 8 links 精确 lumped，此前丢失的 1.023kg 手部质量恢复）。整机 vs Isaac URDF 资产：**ΔCoM 0.0022mm、ΔI 8.3e-6 kg·m²、Δmass −0.002kg**（logo+pelvis_contour 有意跳过）。

**D2 复测**（基线 ANALYTICAL mult=10/kd=8，同 run_fixed12）：

| 指标 | 修复前 | 修复后 | 变化 |
|------|--------|--------|------|
| release | 25.67 | **25.25** | −1.6%（噪声内） |
| PD | 34.33 | **33.83** | −1.5%（噪声内） |
| P1 hip M_eff | 0.08073 | 0.08096 | +0.3% |
| P1 knee M_eff | 0.06437 | 0.06439 | 0.0% |
| P1 ankle M_eff | 0.00962 | 0.00962 | 0.0% |
| rise90 (h/k/a) | 0.155/0.135/0.135 | 同左 | 0.000 |
| 正弦幅值比 (9 组) | — | 1.000-1.056 | ≈1（hip 2Hz 1.056 噪声） |

release 死因 ank_pos=8, ori=3, body_h=2；PD 死因 ank_pos=10, body_h=2（与修复前同分布）。

**判读**：
1. **修复本身验证通过**：整机静态质量属性与 Isaac 资产逐位一致；P1 腿部响应逐位不变（精确 lumping 保持躯干俯仰惯量近似守恒，ixx 0.124→0.1217）。
2. **D1 节的"0.0807→≈0.065"预测被证伪**：torso bug 从来不是髋 M_eff 差距的来源。静态植物差距**关闭**——行为差距不在此。
3. **残留差距重新定位**：armature 表=Isaac 实测表（hip 0.025101925=99.10/ω² 逐位），腿段质量属性逐位一致，驱动定律逐位一致 → 髋 M_eff 的"17-25% 差距"实为 P1 root-fix 设置伪影：Isaac 实测 0.065 低于固定基模型 0.078 达 17%，我们 0.0807 高于模型仅 3.5%——两侧偏离同一模型方向相反；踝三值（0.0111/0.0100/0.00962）互差 ±10% 内。
4. **结论**：单关节动力学、静态整机、驱动定律、armature 全部对齐。剩余 PD 生存差距（34 vs 18.6）必在整机/接触侧——**任务 #3 ② PD 轨迹 diff 为决定性下一步**（同一 12 片段逐帧对比，定位首个分歧自由度）。

### ② PD 轨迹 diff (08-19→20)：同一 12 片段两侧逐帧对比 —— PD 层无差距，剩余差距全部在 release 层

**数据**：D3 收集（24 episode npz：12 release + 12 PD，逐 ctrl_step 记录 29-dof qpos/qvel/ref_qpos/joint_target/applied_torque）+ Isaac P0 侧同字段。两侧 eval 均确定（D2/D3/D4 release 25.25/PD 33.83 逐位一致）。

**对齐前提**（此前已解决）：
- 列序：我们 XML 序 → isaaclab 序用 gather `data[:, ISAAC2XML]`（初始误用 scatter 方向产生过假"重定标不同"结论，已修正）。
- 末行污染：我们 eval 每 episode 最后一行是 post-reset 状态（root_pos≈root_pos[0] 至 1e-8），比较时丢弃。
- Isaac root_pos 含 env origin（3 origins 循环）、ref_root_pos 为 env-local；Isaac 起点 = asset 默认位姿（heading −85°，不做 ref 朝向对齐）；Isaac refs 50Hz 平滑、我们 30Hz step-hold。

**逐项对比结果**：

| 项 | 我们 | Isaac | 判决 |
|----|------|-------|------|
| ref 内容 | — | — | **0.003-0.016 rad 一致**（同片段同相位，目标=ref 逐位）|
| PD 跟踪误差 (|q−target|) | 0.03-0.15 rad | 0.14-0.26 mean，峰值 waist_pitch −0.33 / r_ankle +0.31（软增益下垂）| **我们跟踪更好** |
| root 漂移 | 0.04-0.38 m | 0.13-0.40 m | 我们更小 |
| z-sag | −0.02 ~ −0.06 | −0.04 ~ −0.12 | 我们更小 |
| **生存（同严格阈值）** | **34.33** | **18.58** | **我们 PD 更长寿** |
| 同口径 release | 25.67 | 87.33 | **全部剩余差距在此** |

**判读**：
1. **PD 层（植物+驱动+接触的联合响应）无差距**——同一批 ref、同一批严格阈值下我们的 PD 生存显著高于 Isaac 自身 PD。D2 之后"剩余差距在整机/接触侧"的悬案到此收敛：PD 轨迹对比直接把整机侧也排除了。
2. **剩余差距 100% 在 release 层**（87.33 vs 25.67）：策略观测/输入层（obs 侧注入、指令、增益标度）或策略力矩在我们引擎下的生效差异。这与 release P0 采集 bug（策略实际消费每 env 固定的 3 条动捕）互相印证——release 侧数据至今不可用，同事的 obs 侧修复重采是唯一关键路径。
3. **后续**：applied_torque 读回修复（get_joint_forces 读 eFORCE 注入缓冲 → 全零；已换 get_joint_torques=eLINK_INCOMING_JOINT_FORCE）后做扭矩签名对比，作为植物载荷的最终硬校验（见下）。

### ② 扭矩签名对比 + applied_torque 读回修复 (08-20)

**🔴 发现**：我们 eval 的 `applied_torque` 字段**全零**——`get_joint_forces()` 读的是 eFORCE cache（外部注入缓冲，eACCELERATION 驱动下从未写入）。换用 `get_joint_torques()`（eLINK_INCOMING_JOINT_FORCE，真实现）后 D4 重采（release 25.25 / PD 33.83 逐位复现，物理确定性不受读回影响），12 片段扭矩全部非零（mean 3.48、max 47 Nm）。

**读回正确性验证（决定性）**：τ_rec vs kp·e − kd·v（ANALYTICAL mult=10/kd=8 增益实测：hip 1098.7/558.8、knee 8848/4500、ankle 142500/72000）——**无限幅下 corr = 0.87-1.00 全关节**（腿/腰 0.99+，腕 roll 0.01 Nm 噪声除外）。同一列索引对 qpos/qvel/tgt/tau 同时成立 → 读回顺序 = XML 序，驱动定律端到端逐位执行。**我们的驱动就是公式本身，没有隐藏缩放**。

**跨引擎扭矩对比判读**：
- 增益相差 11-5000×（我们 ankle kp=142500 vs Isaac 28.50），扭矩波形必然不同（我们紧跟踪 vs 他们软增益下垂）——跨引擎逐关节 corr 0.3-0.67、模板匹配 8/29 是增益差异的必然结果，**不是植物信号**。
- 扭矩字段两侧都只测控制努力（τ = kp·e − kd·v，Isaac 侧 ④ 已证 R²=1.00000），不直接测植物载荷；载荷只能从运动响应推断——而这正是 PD 轨迹 diff 已回答的问题（我们跟踪更好、生存更长 = 植物无赤字）。
- **结论**：扭矩层无额外差距。任务 #3 完整闭合：ref 对齐 → PD 轨迹 → 驱动定律 → 扭矩读回，四层全部一致或我们占优；**剩余差距 100% 在 release 层（87.33 vs 25.67），唯一关键路径 = 同事 obs 侧修复重采**。

**顺带发现**：记录扭矩峰值 135.5 Nm（r_knee，crutch 片段）未触限幅——actuatorfrcrange 限幅在 135+ 或未生效，与 Isaac 侧限幅（ankle 10.02 mean 处未见饱和）一致，不阻塞。

### ② release 层本地排查 (08-20→21)：action 语义解密 —— 我们的 scale 是正确的，D6 identity 证伪

**动机**：release 差距（87.33 vs 25.67）100% 在策略输入/输出层。逐一排查 obs 结构与 action 语义。

**Isaac action 语义解密（回归法）**：用 Isaac 数据回归 `action_raw vs ref_qpos`（PD 模式）与 `action_raw vs joint_target`（release 模式）：
- **PD 模式**：`action_raw = 1.0×(ref − offset)`，29/29 关节 slope≈1.0（R²>0.9），offset 与我们 `_ISAAC_ACT_OFFSET` 一致 ≤0.02 rad（膝 0.680 vs 0.669、踝 −0.381 vs −0.363、肘 0.581 vs 0.6）→ **PD 模式 action manager = identity**。
- **release 模式**：`action_raw` 巨大（|max| 5.14-5.24，34% 条目超 ±1），`(target−offset)/action_raw` = **0.35-0.55 逐关节 = 我们的 deploy-stack `_ISAAC_ACT_SCALE`**（指纹：腕 yaw 0.076±0.005 vs 我们 0.0745；l_sh_yaw 0.445±0.029 vs 0.4386；la_pitch 0.438 vs 0.4386）→ **release 模式 = normalized action × scale，我们实现正确**。两种模式 action 语义不同（PD 直通 rad，release 走归一化）。
- 策略为 Gaussian（无 tanh 界）：我们输出 |max| 8.0、Isaac 5.2，同一量级 ✓。

**D6 identity 实验（证伪）**：把 scale 改成全 1 → release **1.33 步**（body_h=7, ori=6 第 1-2 步倒地）。机制：策略输出 ±8 直接当 rad target → target 爆炸 → 被关节限位钳制（D6 step-0 target 全部 = XML 关节限位：lk 1.669、la_pitch −1.363、lh_pitch 0.688…）。D6-PD 30.92 vs D5-PD 33.83 = 浮点路径改变后混沌发散（round-trip 数学恒等，1e-16 扰动在 30+ 步接触动力学中被放大），非系统效应。

**obs_step0 结构核对**（Isaac release env0）：
- 布局确认 [avh 10×3, jph 10×29, jvh 10×29, ah 10×29, gdh 10×3] = 930，与我们一致。
- jph fill 行 = reset 位姿 − offset（逐位），与我们的 fill 修复一致；ah 第 9 行 = 策略第一步 action（逐位 = action_raw[0]）；gdh 第 9 行 = 直立重力 [0.04,−0.04,−1.0] ✓。
- 状态块第 9 行与 npz 记录 qpos[0]/qpos[1] 均不符（最大 0.4 rad），采集时点约定未解——不影响稳态判定。

**下一步**：③b release 重放（A033 500 步 action 注入我们引擎）——同 target 序列下对比 qpos 轨迹：对齐=植物无差距、差距在 obs 环；发散=植物响应 release 类 target 的差距。

### ③b release 重放 (08-21→22)：action clip 根因定位 —— 部署映射在我们的引擎里被 [−1,+1] 裁剪

**重放第一版（A/B 两驱动配置）判读事故**：target/qpos 与 Isaac 全面发散（target mean|d| 最高 1.0 rad、corr 大量负值）。排查发现两个 bug：
1. **分析脚本列序 bug**：`tgt[:, I2X]` 把 isaac 数据按 isaac 位序输出，与 XML 序错位对比（我们 lk vs 他们 rh_roll）——已修正为 `tgt[:, argsort(I2X)]`（XML 标签 gather）。
2. **重放脚本注入 bug（真 bug）**：`ar_isaac[k][ISAAC2XML]` 把 isaac 位序 j 的 action 放进 XML 槽 j（只有 lh_pitch 与 r_wr_yaw 两槽恰好一致），27 关节动作全错位 → 机器人原地踏步、轨迹反相关。正确：`ar_isaac[k][XML2ISAAC]`（XML 标签 gather）。env.step 无内部重排，评测管线始终正确（策略在我们环境训练，天然输出 XML 序）。

**修正后重放 B（FORCE+CFG 真增益）**：
- **target 对齐**：24/29 关节 mean|d| ≤ 0.09 rad、corr 0.86-0.98 → **action→target 映射两侧逐位一致（scale/offset 表正确）**。
- **🔴 5 个离群关节的 target 在我们侧平顶**（la_pitch 65% 时间钉在 +0.0756、l_sh_yaw 100% 钉在 −0.4386、r_elb 72% 钉在 +1.0386、膝 23% 钉在 +0.3184、腕 pitch 80% 钉在 +0.0745…）。**每个平顶值 = scale×clip(±1) + offset 的精确解**（la_pitch: 0.4386−0.363=0.0756；膝: 0.669−0.3506=0.3184；l_sh_roll: 0.4386+0.2=0.6386）→ **physx_env.py step() 里的 `np.clip(action, −1, 1)`（MuJoCo 时代遗留）把动作裁到了 ±1**。
- **Isaac 不裁剪**：他们的踝 target 1.672 rad = 0.4386×4.6 action——Gaussian 策略输出 ±5.2（34% 条目超 ±1），部署映射为纯仿射。**我们的 ±1 裁剪把 target 摆幅上限压到 ±scale（0.35-0.55 rad），只有行走所需摆幅的 1/3-1/4**——策略输出 ±8 全被钉在 ±1。
- 行为后果与重放一致：**我们机器人 500 步原地踏步（位移 0.002m）vs Isaac 横走 0.10m**；qpos 腿关节 corr 0.70-0.83（形态相似）但幅度被裁剪抑制。
- 修复：移除 clip（本地 + NPU 容器）。E7 无裁剪评测 = 87.33 vs 25.67 差距的候选根因验证。

### ③b 无裁剪重放 + E7 评测 (08-22)：clip 移除后 target 逐位一致，但当前策略是 clip 训练产物

**无裁剪重放 B（FORCE+CFG 真增益）**：
- **target 29/29 关节 mean|d| ≤ 0.09 rad、0 个关节超 0.1 rad**——action→target 映射与 Isaac 逐位一致（scale/offset/clip 三层全部对齐，部署语义解密完成）。
- qpos：腿 corr 0.67-0.85（lk 0.778、rh_roll 0.840、rh_yaw 0.854）、腕 0.78-0.94、踝/腰仍低（la_pitch 0.063、waist_pitch 0.188）；初始 qpos 双方一致（≤0.05 rad 全关节）→ 同初值、同 target 下轨迹仍有残余分歧 = 植物对 release 类 target 的闭环响应差距（接触/根动力学侧）。
- 双方根位移都很小（我们 0.05m@−34°、Isaac 0.10m@87°）——该片段策略实际消费的固定动捕不是 walk_sideway（P0 采集 bug），双方都在原地踏步级运动，yaw 漂移 6.3° vs 3.0° 同量级。
- 变体 A（stiff mult=10）反而更差（腿 corr 0.34-0.56、z 最高 0.941 弹跳）——stiff 增益放大动力学分歧，Isaac-faithful 的软 CFG 增益才是正确对照。

**E7 评测（无裁剪）**：

| 配置 | release | PD | 注 |
|------|---------|-----|-----|
| D5（有裁剪） | 26.00 | 33.83 | |
| **E7（无裁剪）** | **1.00** | **30.92** | release 全灭，PD 噪声内不变 |

- **release 1.00 = 灾难性**：策略输出 ±8（56.6% 条目超 0.9）在训练时被裁剪钉在 ±1 是"满量程"语义；无裁剪后 target = offset + scale×8（踝 −0.363+0.4386×8 = 3.15 rad）→ 撞 XML 限位 → 第 1 步全灭（ank_pos=8, body_h=6, ori=4）。
- **判读**：clip 移除对管线保真是正确且必要的（Isaac 部署映射无裁剪，已由他们的 target=scale×4.6 证明），但**当前 release 策略是 clip 语义下训练的产物**——它的输出分布（±8、饱和式满量程）只有在裁剪下才构成有效控制律。87.33 vs 25.67 的差距根因升级为：**训练动作语义错误（裁剪 vs 不裁剪）**，而非单纯的评测侧 bug。
- PD 30.92 vs 33.83 = 噪声内（ref_action 极少超 ±1，裁剪几乎不生效），与预期一致。

**下一步（E8 路线）**：无裁剪重训 release 策略（BC warmup + PPO，physx_env.py 与 mujoco_env.py 的 clip 均已移除，训练与评测语义一致）。重训后残留的植物侧分歧（重放 B 腿 corr 0.7-0.85）将直接体现在新策略的生存步数上——若仍显著低于 87.33，剩余差距就在植物闭环响应（接触/根动力学），需要回③b 变体对照定位。

## Round-2 规格阶梯扫描 (08-19)：🔴 drive type 是头号杠杆，接触层在 FORCE 下才生效

### 背景：round-2 交付 + 规格对比
收到 round-2（B1 obs 注入修复 + A1-A3 规格）。对比我们 vs Isaac 接触层：
| 参数 | 我们 | Isaac |
|------|------|-------|
| 脚几何 | 1 个 box 16×7cm | **7 capsules/脚**（URDF 圆柱运行时转 capsule，r 0.008-0.01）|
| 脚摩擦 | 0.6/0.5（共用）| 0.787/0.531/0.163 |
| 地面摩擦 | 0.6/0.5 | 1.0/1.0 |
| bounceThreshold | 2.0 | 0.5 |
| frictionOffset/CorrDist | 默认 | 0.04/0.025 |
| vel_iters | 1 | 4 |
| dt | 0.001961×10 (50.99Hz) | 0.005×4 (50Hz) |

实现：SONIC_PHYSX_ISAAC_FEET（7-capsule 脚扇 + 脚材质）、SONIC_PHYSX_GROUND_FRICTION、bounce/frictionOffset/corrDist/vel_iters env-var 开关、set_shape_material()、replay --native-dt/--decimation。全部 legacy 默认，逐档可切。（commit 7fd1244）

### ACCELERATION 阶梯（A033 round-2）：接触层全灭
| cfg | leg mean | la_pitch | 注 |
|-----|----------|----------|-----|
| cfg0 默认 | 0.292 | −0.028 | |
| cfg1 +Isaac 脚 | 0.346 | −0.139 | 接触点数 6→19，几何确认生效 |
| cfg2 +地面摩擦 | 0.333 | −0.117 | |
| cfg3 +接触参数+vel_iters | 0.342 | −0.138 | |
| cfg4 +dt 0.005×4 | 0.344 | −0.129 | |

全部噪声级波动 → **接触层不是分歧源（ACCEL 下）**。

### 🔴 关键发现：drive type 是杠杆
- round-1 12-clip 表（ACCEL 默认）leg mean 0.10-0.39 —— **旧"腿 corr 0.67-0.85"表不复现**。对照训练日志：旧表是 ③b 重放 **FORCE+CFG** 跑的（此前会话已用过 FORCE），committed 脚本默认却是 ACCELERATION。
- 同数据 FORCE vs ACCEL：A033 leg 0.464 vs 0.292；A050 0.406 vs 0.081。**FORCE = Isaac 真实语义（τ=kp·e−kd·q̇ + forceLimit 钳位），是 loader docstring 注明的 faithful config；ACCELERATION 是近似。**
- **训练管线（physx_env_manager.py）从未传 drive_type → 全部生产训练跑在 ACCELERATION 上。**

### FORCE 阶梯（A033 round-2）：接触层生效，全规格最优
| cfg | leg mean | la_pitch | ra_pitch | lk | 注 |
|-----|----------|----------|----------|-----|-----|
| cfgF0 FORCE 默认 | 0.464 | 0.227 | 0.407 | 0.768 | |
| cfgF1 +Isaac 脚 | 0.463 | 0.267 | 0.428 | 0.789 | 单加脚=中性 |
| cfgF2 +地面摩擦 | 0.496 | 0.335 | 0.447 | 0.816 | |
| cfgF3 +接触参数 | 0.467 | 0.257 | 0.409 | 0.776 | 接触参数中性 |
| **cfgF4 +dt 0.005×4** | **0.517** | **0.449** | **0.509** | 0.832 | **踝 corr 首次建立** |

- 双方全程直立（root_z 0.70-0.82）。**踝 pitch 从 ≈0（历史最差定位关节）→ 0.45**。
- dt 档单独贡献 +0.05 leg / +0.19 la_pitch（10 子步 50.99Hz → 4 子步 50Hz，Isaac 忠实值）。
- round-1 A033 FORCE 旧值为 la 0.063 → round-2（B1 修复后自洽数据）0.449：**旧数据观测污染造成的低估被修正**。

**下一步**：round-2 12-clip 全规格表（进行中）；E8 生产栈切换 = FORCE + Isaac 脚 + 地面摩擦 + dt 0.005×4（bounce/offset/corrDist/vel_iters 中性，取 Isaac 值保真）。

### round-2 12-clip 全规格表（FORCE + Isaac 脚 + 地面摩擦 + dt 0.005×4）

| clip | leg mean | la_pitch | ra_pitch | lk |
|------|----------|----------|----------|-----|
| A509 big_heavy | 0.407 | 0.495 | 0.506 | 0.728 |
| A518 crutch_turn | 0.556 | 0.304 | 0.171 | 0.657 |
| A516 crutches_arc | 0.553 | 0.370 | −0.181 | 0.785 |
| A078 inj_right | 0.570 | −0.023 | 0.760 | 0.547 |
| A232 injured | 0.557 | 0.361 | 0.304 | 0.649 |
| A338 injured_torso | 0.530 | −0.030 | −0.316 | 0.849 |
| A050 ff_loop | 0.441 | 0.159 | −0.371 | 0.748 |
| A051 ff_stop_270 | 0.601 | 0.315 | 0.323 | 0.868 |
| A418 ff_stop_360 | 0.512 | 0.137 | 0.419 | 0.830 |
| A514 into_door | 0.605 | 0.465 | 0.160 | 0.827 |
| A033 sideway | 0.510 | 0.375 | 0.589 | 0.833 |
| A476 the_dog | 0.412 | −0.395 | −0.076 | 0.653 |

- leg mean 0.41-0.61（均值 ≈0.52），全面高于 ACCEL 表（0.10-0.39）。膝 0.55-0.87 稳健。
- **踝 pitch 仍是残留分歧关节**：4 条片段踝 ≈0 或负（A078 la −0.02、A338 −0.03、A476 −0.40、A050 ra −0.37）——下一步候选：踝驱动/惯量（Isaac 踝 M_eff 0.0111 vs 我们模型 0.0100）与根动力学。
- 裁判口径与旧表一致（heading 未校正的原始 qpos corr、ISAAC_REORDER 映射、wrist 除外）——0.52 是新底数，E8 训练在该栈上进行后应以生存步数直接验证。

### 🔴 同事发现 #3：replay 初始相位污染 (08-19) —— 全部新表绝对值被低估

**机制**：committed replay 脚本没设 `_forced_ref_time`，env reset 相位随机采样。他们 a[0] 是策略针对**自己帧 0 姿态**的小修正（post-step 仅 0.067），作用在我们随机相位姿态上 → 首步暴力拽动（fd 速度 24.6 rad/s vs 他们 3.0）→ 整条轨迹从第 0 步带瞬态垃圾（root z 蹦 22cm：0.749-0.971，本轮 sweep 中已观测到，当时误判为 clip 行为差异）。

**旁证（本侧）**：旧 ③b 重放 B（相位匹配，初值差 ≤0.05 rad）FORCE 腿 corr 0.67-0.85 vs 本轮随机相位同数据 0.46 —— 污染量级 ~0.2。

**影响判决**：FORCE vs ACCEL 排序、12-clip 相对顺序仍成立（同受污染，FORCE 显著更高）；**绝对值全部低估，踝逐关节判决与 dt 档 +0.19 贡献需 clean replay 定论**；cross_eval 零样本不受影响（随机相位是训练分布内行为）。

**修复**（同事侧已备好，授权推送容器）：release_replay.py 补丁 = 强制相位 0 + plant 覆盖他们 ref_qpos[0]/ref_root[0]（xyzw→wxyz）+ qvel=0 + 目标同步 + --legacy-reset A/B + --native-dt/--decimation；replay_compare_r2.py 复核。推送后 clean replay 复核 A033 阶梯。

### FORCE 零样本 2×2 (08-19)：FORCE/ACCEL × release/PD @ 全 Isaac 栈

- 配置：胶囊脚 + 地面摩擦 1.0/1.0 + dt 0.005×4 + vel_iters 4，24 eps × 0.35/0.35，isaac space，无裁剪。
- 历史锚点：FORCE 17.33/11.58（08-17 带 clip）、ACCEL 25.67（clip）/1.00（E7 无裁剪）、PD 27.17-30.92。
- 目的：D1 悖论裁决（corr 证据 FORCE 优 vs 生存证据 FORCE 差，D1 判定基于已修复的 torso 质量 bug/action clip/伪影增益）+ E8 前"release 权重在已对齐环境能跑多远"新底数。

### FORCE 零样本 2×2 结果 (08-19)：D1 悖论裁决 = 两个域各自成立

| 配置 | release | PD | 死亡模式 |
|------|---------|-----|----------|
| FORCE | **5.25** | 27.88 | rel: ank 22/24；PD: ank 14 + body_h 10 + height 4 |
| ACCEL | 1.33 | 42.29 | rel: ank 17 + ori 10；PD: ank 22 |

（全 Isaac 栈：胶囊脚+地面摩擦+dt 0.005×4+vel4，无裁剪，24 eps @ 0.35/0.35）

- **release 双路皆死**（5.25 vs 1.33）：clip 训练权重在任何无裁剪环境不可运行，E8 重训是硬前提。5.25 = 已对齐环境新底数。
- **D1 悖论裁决**：corr 域 FORCE 优（0.52 vs 0.25）、生存域 ACCEL 优（PD 42.29 vs 27.88）——ACCEL ×15000 刚度"扶住"机器人（生存长、物理错），FORCE 忠实软驱动跟踪准、下垂多。不是悖论，是驱动刚度的固有 tradeoff。
- **FORCE PD 27.88 > Isaac 自侧 PD 18.58**：我们 FORCE 栈在 PD 生存尺度不比 Isaac 严苛——E8 在 FORCE 栈训练不会被物理侧拖死。
- **接触栈作用域**：ACCEL PD 30.92→42.29（+37%），FORCE PD 27.17→27.88（噪声）——接触栈对生存的帮助在 ACCEL 域生效、FORCE 域中性。

**E8 生产栈（待 clean replay 复核后实施）**：FORCE + Isaac 脚/摩擦 + dt 0.005×4 + vel4，BC warmup + PPO，训练随机相位不变。

### clean replay 复核 (08-19 晚)：相位污染真实但 500 步影响 <2%，clean 表 ≈ 污染表

**clean FORCE 阶梯（A033，state-override）**：cfgF0 0.476 / cfgF1 0.450 / cfgF2 0.484 / cfgF3 0.504 / cfgF4 0.494 —— 与污染版（0.464/0.463/0.496/0.467/0.517）噪声级一致。**dt 档 +0.05 优势在 clean 下消失（cfgF3 0.504 vs cfgF4 0.494）→ dt 0.005×4 是中性项（保留 Isaac 值）**。

**clean 12-clip 表**：均值 0.525 vs 污染版 0.519，逐片段 Δ ≤ ±0.02。首步瞬态（fd 24.6 rad/s、root 蹦 22cm）实测存在，但对 500 步 corr 贡献 <2%。**同事"绝对值全部低估"预期未获数据支持；"排序成立"成立**。配对口径（shifted vs same-index）差异可忽略（0.494 vs 0.495）。

**两个确定收获**：
1. **fixed12 pkl 悬案关闭**：|reset_qpos − override| = 0.005 rad —— 我们 pkl 与他们录制 ref 帧 0 几乎逐位一致（旧估计 ≤0.05）。
2. **踝分歧模式 clean 下原样保留**：A476 la −0.35、A050 ra −0.37、A516 ra −0.16、A338 踝 ≈0 —— 踝 pitch 是真实植物残留分歧（非相位伪影），为 E8 后第一定位目标。

**生产栈切换最终论据**（同事门槛"clean 数字上升"未满足，但切换理由不变）：FORCE 忠实语义（corr 0.52 >> ACCEL 0.25）+ FORCE PD 27.88 > Isaac PD 18.58（我们栈不比 Isaac 严苛）。E8 = FORCE + Isaac 脚/摩擦 + dt 0.005×4 + vel4，BC warmup + PPO。

## 🔴 obs_step0 取证 (08-19 深夜)：actor obs/动作关节序与 Isaac 错位 —— release 层差距头号根因

**方法**：round-2 B3 的 obs_step0 是 reset 时刻捕获（ah 全零、jph 10 帧静态）→ 指纹法失效。改直接 obs-vs-obs：我们在 NPU 以同样 reset 条件捕获（root quat 逐位相同 [0.7397, 0.0109, 0.0064, −0.6729]）。

**证据**：
- jph 块 as-is diff 0.786 → **按 isaaclab→mujoco 重排后塌缩到 0.02** → 他们 obs jph = isaaclab 序，我们 = XML/mujoco 序。jvh/ah 同错位。
- 动作通道同错位（代码确认 `_pd_control` 无重排）：release 策略输出 isaaclab 序，env.step 按 XML 序消费 → 策略的膝关节命令打在我们右膝上（slot 9 是 isaaclab lk ↔ XML rk）。
- 08-17 obs 审计（6eabb5e）修了块顺序/相对化/cmd 序，**actor 块内部关节序从未被检查**——"obs 不是瓶颈"（release 25.08）结论建立在错位 obs 上。
- gdh/jvh 差异 = 他们 reset 后 settle 捕获（有速度）vs 我们静止——捕获时机语义差，次要。

**影响**：release 策略在我们环境关节身份全面错乱（只有 ~4/29 槽幸存：slot0/2/6/12…）→ 87 vs 26 差距、±8 vs ±5.2 放大环、E7 灾难的头号新嫌疑。修复方向：physx_env/mujoco_env 输出 isaaclab 序 actor obs + env.step 接受 isaaclab 序动作内部转 XML（env 对外 = Isaac 部署语义），修后重测 release 零样本 + BC warmup 重跑。

## P1 踝 FORCE 专项 (08-19 深夜)：驱动层无分歧 + Isaac P1 时间轴 4× 压缩定量钉死

- 我们 rise90 **135ms** ≈ CFG 增益理论值 128ms ✓；Isaac 标注 40ms = 理论 3.4× 快。
- **决定性证据**：他们 "5Hz" 正弦幅值 0.909 = 理论在 **1.25Hz** 的响应 0.91；我们 5Hz 实测 0.475 = 理论 5Hz 0.479 → dt 标签 0.005/实际 0.02 的 bug 精确证实（4×）。
- 修正后他们真值 ≈160ms vs 我们 135ms（略快略硬）→ **踝驱动/惯量无分歧**（armature 维持 D1b 结论 ON）。踝 replay corr 残留（0.45，绕过策略的纯植物层）仍需定位，但排除驱动层。

### joint-order gate A/B (08-20)：obs/动作序修复生效 +132%，但非充分 —— 剩余差距在植物闭环+clip 语义

**实现**（同事补丁，commit 187a668）：SONIC_PHYSX_ISAAC_JOINT_ORDER 门控（默认 ON）——actor/critic jph/jvh/ah 按 [:, ISAAC_REORDER] 发射 isaaclab 序；step() 在 trust 混合前 action[ISAAC2XML] 转 XML；ref_action BC 目标 isaaclab 序。OFF = legacy XML 接口。植物零改动（纯接口置换，gate ON/OFF 轨迹逐位一致烟测通过）。

**A/B 首格（FORCE + Isaac 接触栈，24 eps @0.35）**：
| 格 | 结果 | 判读 |
|----|------|------|
| A gateON release | **12.17**（ank 13, ori 11, height 2）| +132% vs OFF；ori 死亡 5→11 = 策略拿回正确关节身份、真正尝试控制 |
| C gateOFF release | 5.25（ank 22, ori 5）| 逐位复现 2×2 基线 → A/B 干净 |
| B/D PD | 待补 | sanity：应对 gate 不敏感 |

**判读**：obs/动作序是必要项（灾难级 5.25 → 12.17），**非充分项**——距 87.33 仍有 7×。剩余候选：① clip 训练语义（无裁剪 target 撞限位）② 植物闭环残留（踝 replay corr 0.45、root 动力学）③ obs 层遗留（mujoco 块序、SMPL 块、噪声注入）。E8 = gate-ON 环境无裁剪重训仍为主线。

**A/B 全表（08-20）**：

| 格 | gate | 模式 | 结果 | 死亡 |
|----|------|------|------|------|
| A | ON | release | **12.17** | ank 13, ori 11, height 2 |
| B | ON | PD | 27.88 | ank 14, body_h 10, height 4, ori 1 |
| C | OFF | release | 5.25 | ank 22, ori 5 |
| D | OFF | PD | 27.88（逐位同 B）| 同 B |

- B/D 逐位相同（27.88 = 2×2 复现）→ **gate 对 PD 路径零影响，sanity 通过**；C 复现 5.25 → A/B 干净。
- A vs C +132%：修复生效但非充分——ori 死亡 5→11 = 策略拿回正确关节身份后在真正控制，踝位置漂移（ank 13）仍是第一死因，与踝 replay corr 0.45 的植物残留互相印证。

## ⚠️ 归因更正 (08-20)：撤销"clip 训练语义"——release 权重从未在裁剪下训练

- **错误归因（E7 时代）**："release 策略是 clip 语义下训练的产物"——不成立。release 权重是 NVIDIA 在 Isaac 无裁剪训练的（他们部署同权重 target 可达 scale×4.6 > ±1）。
- **正确机制**：裁剪是我们评测侧遗留物，其真实作用是安全阀掩盖 obs 关节序错位（错位把策略推到 ±8 vs Isaac 侧 ±5.2）。gate-ON 修复 obs 序后，clip 不参与当前差距解释。
- **修正后归因**：release 10.62 vs 87.33 的剩余差距 = 植物闭环残留（踝 replay corr 0.45 纯环境量）为主 + obs 遗留项（SMPL/噪声注入/settle 语义）。
- E8 理由修正：不是"纠正训练语义错误"（无此错误），而是"release 权重在新修复环境仍不能跑 + VLA 底座需自有环境训练 + round-trip 检验需要 E8 策略"。无裁剪训练保持与 Isaac 部署语义一致。

## 踝机制专项 (08-20)：sep 之谜降级（报告层伪影）+ 接触负载模式首测

**sep=+0.039 裁决**（队友判据）：回调捕获 96 点/120 步（81 步零点），sep 恒 0.0395、impulse 全零——但**接触物理真实**：踝传递扭矩支撑时刻超驱动需求 13-19 Nm（k=50: 14.57 vs 1.07；k=90: 18.88 vs 2.79），尖峰 150 Nm 超 50 Nm forceLimit（非驱动成分）。→ sep 之谜 = 回调层伪影，降级。**回调只捕获单一 actor pair 的 speculative 点，真实接触对漏记**。

**onset 分析**（4 坏片段 + A033 参照，clean replay 本地）：踝跟踪误差双方都大且集中在支撑相（我们 stance 0.35-0.84 vs 摆动 0.07-0.29；他们 0.35-0.71 vs 0.05-0.30）——**软踝驱动（CFG kp=28.5）承重欠跟踪是设计行为，两引擎同款**。坏片段轨迹反相关 + 误差同量级 → 弱驱动踝由接触反力主导，反力模式差异驱动轨迹分歧。

**接触负载模式首测**（我们踝轴接触扭矩 vs 他们 contact_force_left/right 剖面）：corr 0.02-0.19（支撑相 −0.12~0.18）——初步"模式不同"信号，**口径 caveat**：轴投影扭矩 vs 总力向量量纲不对应；我们 readback 接触分量仅尖峰可见。**阻塞项：接触回调漏记真实接触对 → 修回调是接触层直接测量的前提**。

## 接触回调修复 + 负载对比 (08-20)：站立捕获 100%，行走漏 40%，L/R 失衡候选

**回调根因**：filter shader 只设 TOUCH_FOUND/LOST 无 PERSISTS → 回调只在触地/离地瞬间触发（96 点/120 步、sep 恒定、impulse 零）。修复（commit 3e8616c）：+eNOTIFY_TOUCH_PERSISTS + 接触点世界坐标进 debug tuple（尾部追加，旧索引不变）。

**验证**：站立 hold 1s 捕获冲量 332 vs 重力需求 334（比值 1.00，100 点/步）；行走态 52 点/步、冲量 63%（缺 40%）；pairs 仅 2 个 distinct actor pair（14 胶囊应 14 对）——多数 link 接触疑以 articulation 身份塌缩报告。**全足底 x 范围覆盖，非空间缺口。**

**GRF 对比（体坐标系分类，z 分量）**：我们 L 160/R 28（A033）、L 122/R 29（A050）vs 他们 L 196/R 180、L 183/R 190——**我们左右负载严重失衡（左重），他们均衡**。剖面 corr −0.02~−0.30（支撑相 −0.18~0.04）。⚠️ 失衡是候选发现，待行走态捕获补全后定论。

## 🔴 L/R 负载失衡确认为真发现 (08-20)：左脚承重+右踝撞击 = 跛行模式

**裁决方法**（gr00t-wholebodycontrol-89 审核建议）：踝扭矩读回（get_joint_torques，corr 0.87-1.00 已证）+ root roll 佐证，无需新捕获。

| 片段 | 我们踝扭矩 L/R 均值(Nm) | root roll | 他们足力 L/R(N) |
|------|------------------------|-----------|------------------|
| A033 | 5.8 / 1.1（5.3×）| +3.9°（左倾）| 196/180 均衡 |
| A476 | 6.8 / 2.1（3.2×）| +2.3°（左倾）| 191/176 均衡 |
| A050 | 5.2 / 1.8（2.9×）| −0.6° | 183/190 均衡 |

- 左重右轻 3-5× + root roll 同向左倾（2/3 片段佐证）；右踝偶发暴力尖峰（147.9 Nm，3× forceLimit——冲击）而均值近零。
- **画像：左脚持续承重、右脚间歇撞击 = 跛行/原地踏步模式**，与 clip 时代"500 步原地踏步（位移 0.002m）"吻合。同动作同目标下 Isaac 双侧均衡交替 → 纯植物层差异。
- **意义**：首个把植物差异直接接到死亡模式的候选——左倾 → roll 漂移 → ori 死亡 + 右踝撞击 → 踝轨迹分歧，同一机制解释两大残留。GRF 捕获缺陷（行走态缺 40%）修复后做最终定量确认。

## 跛行机制定位 (08-20 续)：bounce 排除、失衡恒定、下一步镜像扰动

- 支撑相分解（他们力作 stance 参照）：A033 右支撑相踝扭矩 0.4 Nm vs 左 7.3（16×）；A050 0.6 vs 3.9（6.8×）；A476 2.2×。**右腿落地不承重，重量恒定压左腿**（100 步块比值恒定 15/21/15/15，非正反馈）。
- roll 时间序列：A033 左倾 0.5-7.8° 振荡、A476 恒定左倾、A050 ≈0 但仍有 6.8× 失衡 → **roll 是伴随现象非起因**。
- 右踝尖峰：A033 的 3 尖峰聚在一次触地（449-451）；A476/A050 散布。
- **bounce 假设排除**：0.5 vs 2.0 下 L/R 比逐位相同（16.3、尖峰 3）——触地弹回不是机制。
- 画像：对称物理 + 弱横向镇定 → 初始瞬态选边（审核方分岔点框架）。下一步镜像扰动测试：初始 root 横向 ±2cm，承重侧是否随初始条件翻转。

## 🔴 跛行机制定位完成 (08-20 深夜)：COM 横向振荡缺失（根动力学）

**排除链**：bounce 0.5 vs 2.0 逐位相同 → 初始 root 横向 ±5cm 全部无效（强吸引子，非初始条件选边）→ 髋 roll 跟踪无差异（我们 0.17-0.22 ≈ 他们 0.17-0.22）→ FORCE 驱动是 solver 原生弹簧（stiffness/damping 每子步求解，无 ZOH）→ 右支撑相右脚确实贴地（70-84%，min 0.035，相位正确）。

**定位**：COM 横向振荡缺失——他们 root 横向位置在 R-stance vs L-stance 差 **0.23m**（A033 侧行）/ 0.03m（A050 前行）；我们 **0.001m / 0.014m**。双脚按时落地、髋角度跟踪正常，但重量横向转移不发生，COM 粘在左腿上 = 跛行。

**意义**：① 首个"植物差异 → 死亡模式"的完整机制链（左倾 + 右腿空载 → ori/ank 死亡）；② 与 ③b"根位移 0.05m vs 0.10m"残留同源，首次定量定位到根横向动力学；③ 候选：reduced-coordinate solver 根-腿质量耦合（静态逐位 ≠ 动态等价）或支撑腿准静态平衡差异。下一步：单脚站立标定 + 根横向脉冲响应测试（把"根横向响应弱"变成可修的量）。

## 三场景分叉 (08-20 深夜)：stance 相位内关节偏差是运动学根因，源头回到 GRF

- sc1（植定+根不动）≈0 窗；sc2（脚随根拖）1-2 窗；我们模式 = 整机小幅漂移（足/根 dy 2-9cm），他们 = 根跨过植定脚 20-50cm。
- **运动学定理裁决**：根运动由 qpos 确定 → 相位分辨 |dq|（他们 stance 窗内）：A033 踝 pitch 0.14-0.15 / hip_roll 0.11-0.14；A050 ra_pitch **0.258** / la_pitch 0.184 / rh_roll 0.135 rad——整 clip 均值藏掉的 6-15° 相位内偏差。
- **链条修正**：stance 关节跟踪偏差（我们 > 他们）→ 运动学确定根轨迹 → 横向跨越 23cm→3cm → 重量不过腿 → 跛行。驱动忠实弹簧下，stance 偏差 = 接触载荷扰动的响应差 → 最后一环回到 GRF 模式对比（依赖 ③ 捕获完整性）。
- ④ 补完：SONIC_PHYSX_ROOT_Z_OFFSET 已走 manager env-var 路径（此前对训练无效），e8_launch.sh 钉 0.02，BC smoke 加 rz 0.04 对照组。

## 载荷-偏转 + 跨越分解 (08-20 深夜)：机制是耦合平衡，破环点在首次支撑 GRF

- **跨越分解（FK 混合）**：他们关节+我们根 → 跨越完全恢复（0.31/0.36/0.49 vs 他们 0.31/0.50/0.45）；我们关节+他们根 → 跨越完全消失（0.12/0.15/0.18 vs 我们 0.12/0.14/0.17）。**缺失 100% 关节侧、0% 根侧**——"根动力学"候选出局。
- **载荷-偏转传递函数**：两边都服从 τ≈kp·e（他们斜率 27 ≈ kp 28.5）；他们支撑踝偏转 0.47-0.60 rad/扭矩 12-16 Nm（被载荷拖离目标 28°），我们 0.15-0.26 rad/1-6 Nm（右支撑相 1.1 Nm，几乎无载）。**传递函数相同，差异纯在载荷 = 接触侧**。
- **综合**：关节相位偏差 ↔ 载荷不对称互为因果的耦合平衡。初始状态逐位相同 + 扰动无关 → 破环点在第一次支撑相的双引擎 GRF 响应差。③（捕获完整性）是唯一关键路径，下一步照排：单脚站立标定 + contactCount 日志。
- 附带发现（rz 修复时的历史）：manager 旧默认 root_z_offset=0.0（评测侧 0.02-0.04）——与"训练跑 ACCELERATION 默认值"同源：训练栈从未被显式钉过，现已全部钉入 e8_launch.sh。

## 🔴 P0 接触生成缺陷 (08-20 深夜)：teleport-后-接近场景自由落体穿地

**drop/impact 测试**（审核方建议的破环点标定）：锁腿姿态 + 抬高 3cm 释放：
- root_z 从 0.812 自由落体至 0.32（**z(t) 与 ½gt² 逐点吻合 = 零接触力**），穿地 47cm 后才生成首个接触（3 m/s 深穿透时）
- 逐位复现于：胶囊脚/box 脚、fresh/pre-created pair、bounce 0.5/2.0——**与脚几何、pair 创建、弹回参数均无关**
- 对照：站立 hold（初始置接触高度含 5mm 微穿透）接触正常 100%；行走 replay（驱动控制触地速度慢）正常——**缺陷场景 = 初始高于接触面后的自由接近**

**意义**：第一次支撑相的种子候选——若真实步态中任何触地是快速无控制接近（驱动追赶目标时的瞬态），接触不生成 → 深穿透 → 与 Isaac 的触地响应完全不同。行走 replay 中驱动控制掩盖了它，但种子场景正是它。

**下一步**：最小复现（裸 binding 单 box 从 3cm 落下：穿地=场景级/broad-phase 配置问题；正常=articulation 特有）→ 定位 C++ 层。

## 隧道缺陷二分 (08-20 深夜)：pair 创建失效假设，生产路径不受影响

**分叉表**：env.reset（pkl 值，rz 0.04/0.08/0.12 全高度）✅ / env.reset+手动 simulate ✅ / override（npz 值）+3cm（任意顺序、wake、resetFiltering、单双 teleport）❌ 全穿。裸 box ✅、单 link RC ✅。

**关键观察**：reset 路径的接触在 3-4cm 间隙就出现（sep 0.035-0.04 = 旧 sep 之谜的量级！）→ stale pair 特征。

**假设**：scene-add 时刻（构建姿态，足部在/低于地面）pair 建立且永不溶解 → 所有正常路径走旧 pair；新 pair 创建在接近时失效（大位移重插入才触发 = 47cm 隧道）。override 路径的穿地与此一致。

**实践影响**：生产路径（env.reset）不受影响；缺陷潜伏于"无旧 pair 的全新接近"（跳跃/重建后快速触地）。clean replay 的 override 因初始预穿透幸免。下一轮：C++ 两 box 接近 pair 创建测试 + loader 构建姿态确认。

## ✅ "47cm 隧道"根因落定 (08-20 深夜)：quat 约定翻转 + 睡眠态 teleport 被忽略——broad-phase 无罪

**完整机制**（commit 8663f2a）：
1. round-2 npz 的 `ref_root_quat` **本就是 wxyz**（与我们的 pkl reset quat 逐位相同、yaw 主导旋转 sane）——replay 补丁的 xyzw→wxyz 转换把它翻成 121° 侧躺。
2. 旧 binding `setRootGlobalPose(autowake=false)` 在睡眠 articulation 上**静默忽略 teleport**——clean replay 的 override 从未生效（初始态一直是 env reset 的 0.823 姿态），顺带掩盖了翻转。
3. 我改 autowake=true 后翻转开始生效 → drop 探针把机器人 teleport 成侧躺 → "自由落体穿地 47cm" = 侧躺机器人下方没有任何部件。**broad-phase/pair 创建全部无罪**（spawn-above、同姿态 teleport 一直正常）。

**修复**：① 脚本去翻转（直接 wxyz）；② binding autowake=true（teleport 语义正确 + 顺带修复训练路径上 reset-after-sleep 的静默失效）。

**验证**：A033 clean replay 初始态逐位贴合他们 post-step-0（quat [0.7397,0.0289,0.0208,−0.672]、z=0.7805 稳定），corr 0.498（la 0.363 微升）。12-clip v3 表重跑中。

## v3 12-clip 表 (08-20 深夜)：修正初始态基线，均值 0.529 ≈ v2 0.525

quat/autowake 修复后的正式基线（override 真正生效，初始态逐位贴合 Isaac post-step-0）：

| 片段 | v3 leg | 关键踝 |
|------|--------|--------|
| A051 | 0.605 | la 0.428 |
| A514 | 0.595 | la 0.449 |
| A078 | 0.588 | ra 0.818 |
| A516 | 0.571 | ra −0.202 |
| A518 | 0.554 | la 0.209 |
| A338 | 0.535 | la −0.037 ra −0.324 |
| A418 | 0.513 | ra 0.423 |
| A232 | 0.501 | la 0.297 |
| A033 | 0.498 | la 0.363 ra 0.537 |
| A509 | 0.478 | la 0.559 |
| A476 | 0.463 | la −0.306 |
| A050 | 0.451 | ra −0.383 |

- 均值 0.529 ≈ v2 0.525（初始态修正 ±0.05 内移动，结论不变）。
- **踝分歧模式与所有修复正交**（A338/A050/A476 的踝 ≈0/负 原样保留）→ 踝残留确认为真实植物差异，是 E8 后第一定位目标。
- 基线锁定：**FORCE + Isaac 接触栈 + isaaclab 序 + 修正 teleport，12-clip leg mean 0.53、踝 0.45**。

## 标签修正 + 约定来源 (08-20 深夜，审核方清理清单)

- **旧表重贴标签**：所有 v3 之前的 clean replay 表（含 0.525 均值和 v2 12-clip 表）= **reset-state 初值**（root z 0.823 高 4.3cm、qpos Δ≈0.005），不是"匹配初值"——override 因睡眠态 teleport 被忽略而从未生效。**v3（均值 0.529）是首个真基线**。
- **零样本 A/B（12.17/10.62）不受影响**：cross_eval 走 env.reset（pkl）路径，不经 replay override；死亡时 articulation 在动未睡。"必要非充分"判决数据无需重加权。**初始分离审计作废**（隧道缺陷是 probe 伪影，无缺陷可触发）。
- **约定来源**：08-19 的 xyzw 判决（yaw≈0.4° 证据）针对的就是 round-2 npz——证据是误读，翻转补丁从出生就错。round-2 ref_root_quat = wxyz（与 pkl reset 逐位相同）。录制侧要求：**永远导出 wxyz + npz 元数据盖章约定**（已列入给队友的下次需求）。
- **睡眠态 teleport 的真污染在训练路径**：max-length 终止的站立 episode（睡死）→ reset 被忽略 → 下一 episode 从旧姿态起步。E8 清单新增：训练容器必须带重编译 .so（build 钉版）；BC smoke 加 asleep-reset 对照（站到入睡 → reset → 验证姿态应用）。

## 踝偏转比散点 (08-20 深夜)：坏片段 = 我们踝偏离 ref 2-3.7×（ratio 0.27-0.64）

逐 clip 支撑相 |q − ref_qpos|（他们 npz vs 我们 v3 replay）比值 vs 踝 corr：
- 坏片段：A338 0.27/0.36、A050 ra 0.42、A476 ra 0.42、A516 ra 0.52、A418 ra 0.29——**我们踝偏离 ref 1.5-3.7× 于他们**
- 好片段：A509/A514/A033/A232 ratio 0.87-1.11——两边都贴 ref
- 斜率 +0.36/+0.33（审核方预测负斜率未现）；A509 例外点按预测出现（ratio≈1 + 高 corr）
- 机制修正：他们的踝被载荷拖离 target 0.5 rad 却仍贴 ref（载荷主导的欠控）；我们的踝在坏片段直接偏离 ref 2-3×（轨迹走形）——两种欠控模式。
- **单脚标定的目标量：把坏片段支撑相的踝偏转比压回 ≈1。**

## 踝位置误差分解 (08-20 深夜)：近端 FK 遗传 99-100%，踝本地健康——受害者非元凶

坏片段（A338/A050/A476）+ A509 对照，支撑相踝世界位置误差分解（我们的髋/膝 + ref 踝过 FK vs 全我们的 q）：
- 坏片段总误差 0.12-0.30m，**99-100% 由近端（髋/膝）误差 FK 遗传**，踝本地残差 −2%~+1%（A509 同为 ≈0）。
- 按审核方分叉判据：踝物理健康 → 单脚标定预期干净通过 → **比值修复必须走上游**（接触 GRF → COM 横向 → 髋 roll），动踝参数无效。
- 完整因果链闭合：接触载荷缺失 → 横向转移不发生 → 近端链走形（0.1-0.26 rad）→ 踝 FK 遗传 → 根轨迹错 → 跛行 → ori 死亡。踝是受害者。
- 明日重心：drop 瞬态（链头直接探测器）+ 147.9 尖峰真伪；单脚标定仍做（确认踝本地健康，防止链头修复后踝侧暴露第二层问题）。

## 明日三测试定位图 (08-20 深夜，审核方确认)

链头两段与测试的对应：
- **(a) 首次接触瞬态**（冲量/反弹/摩擦建立）→ **drop 测试直探**。判别力清单：修好的回调（PERSISTS）逐子步录冲量时间序列——峰值冲量、上升时间、settle 形状三项，对照他们已录制的 contact_force 首次支撑相窗口。链头旋钮 = bounce 0.5 / frictionOffset 0.04 / corrDist 0.025，瞬态形状指出哪个参数在哪段分叉。
- **(b) 静立相横向 GRF 建立** → **单脚标定直探**。drop 干净 → 分歧必在 (b)。
- 147.9 尖峰 v3 复核：与隧道/踝侧双重解耦后，仍在 = 链头间歇接触现象（值得查），消失 = 关闭。
- 单脚标定剩余价值：确认踝本地健康 + 防链头修复后踝侧暴露第二层问题。

## drop 瞬态判别 (08-20 深夜)：接触建立健康，链头 (a) 干净 → 分歧在 (b)

修正协议下首测（逐子步冲量，判别力清单版）：
- 接触高度 0.783 精确、上升 5ms、峰值 3764N（=34kg 锁腿自由落体刚性冲击，物理预期量级）、settle 5ms 到 328N（≈体重）、bounce 0.5/2.0 无差异（1 dip 相同）。
- 他们的首触地（120-260ms 上升、431-686N）是驱动控速落地——峰值不可比，形状可比：我们冲击吸收 5ms = 机制健康。
- **三角定位：链头 (a) 干净 → 分歧必在 (b) 静立相横向 GRF 建立 → 单脚标定为判别性测试**。
- 147.9 尖峰复核：失衡存活（16.3→14.6 真植物现象）；尖峰 3→1（2/3 teleport 伪影，剩 1 个待查其触地时刻）。
- 下一步：单脚横向标定（lean-hold + 横向 GRF 测量，链头 (b) 直探）。

## 单脚横向标定 (08-20 深夜)：侧向 GRF 恒零 = CoP 主导，链头 (b) 健康，分歧收敛到近端链

- lean-hold 扫描（0-0.30 rad）：侧向 GRF **所有倾角精确为零**（期望 16.7-98.6N）；法向力正常（335N≈体重，分布平衡）→ 植物用 **CoP 前移到脚尖**平衡倾覆力矩（物理自洽）。
- 他们的录制 contact_force x/y=0（纯法向）——**两引擎同款 CoP 主导模式**，侧向转移的真实机制 = 足部落点摆动力学（纯运动学）。
- 三角定位闭合：链头 (a) 接触瞬态健康、(b) 静立锚定健康 → **分歧收敛到近端链（髋/膝）stance 相位偏差本身**，与 FK 混合"100% 关节侧"、load-deflection"阻抗无差"一致。
- **修复方向**：髋/膝在承重下的驱动-载荷耦合（stance 相位跟踪），不再是接触层。E8 前植物侧的最后一块拼图 = 近端 stance 跟踪的驱动侧定量（P1 式承重响应测试）。

## hip/knee load-deflection (08-20 深夜)：现数据判不动，free-root+植定实验为 gate

- 支撑相偏转量级相似（我们 0.08-0.21 vs 他们 0.09-0.22）→ 承重下我们的跟踪不显著更软。
- 他们斜率 46-96≈kp（驱动定律成立，平凡）；我们斜率≈0 = transmitted readback 被接触反力主导，**现有通道隔离不了我们的驱动输出**（applied_torque 字段语义两侧不同：他们=驱动输出、我们=传递力）。
- 结论：驱动-载荷耦合的定量需要 free-root+植定+CFG 软增益实验 → Round-3 采集请求已起草（docs/physx-isaac-request-r3.md：P1 协议改 free-root+植定脚，其余不变；quat 约定盖章 + 全维 obs 顺带问）。
- E8 时序（审核方建议）：config 侧清单与承重实验并行；BC smoke + PPO 以承重判决为 gate（null → 放行；真信号 → 先修再训）。

## 🔴 R3 判决 (08-20 深夜)：真信号——COM 动力学缺失在矢状面复现，E8 gate = 先修再训

- 数据：同事 R3 交付（commit a9dbd8e，27 组 + 截断版 + 元数据盖章）；我们同协议复现（50Hz 控制步记录、q0 从 npz 内嵌字段、free-root、PD 持姿）。
- **正弦激励根俯仰发散**：他们 9 组全部 d_pitch 0.36-0.37 rad（前倾 21°→2.3s 截断）；我们全部 −0.01~0.00（不动）。**同一软增益/free-root/扰动下，Isaac 植物把髋扰动转化为 COM 俯仰振荡并发散，我们 COM 几乎无响应**。
- 阶跃：他们髋 pitch rise90 1.0-1.8s（承重下极慢），我们 0-0.34s。
- **判读（预注册 fork）**：响应分歧 + 方向一致（我们 COM 抑制）→ 真信号 → **先修再训**。修复目标 = 根动力学/COM 耦合——与整条链一致（横向 25× + 矢状 0.37 vs 0）。
- 下一步：载荷侧分解（我们侧足力记录补一版 → 接触侧 vs 传递侧）；根动力学专项（COM 对扰动/承重的响应传递函数）。

## R3 深读 (08-20 深夜)：稳定性画像相反——我们快响应→CoP 崩塌，他们慢响应→摆模式发散

- **阶跃崩塌**：我们 0-0.9s 稳定 → 1.0-1.2s 下沉 → **1.3s 崩塌**（z 0.55→0.22，1077N 撞击=身体砸地）；他们下沉 45.5mm 存活。**我们 rise90 快 3-5×（0-0.34s vs 1.0-1.8s）→ COM 迅速推到脚尖胶囊边缘 → CoP 饱和 → 倾覆**；他们慢响应 COM 到不了临界点。
- **正弦反向**：我们快响应抑制振荡（d_pitch≈0）；他们慢响应让摆模式振起（0.37 发散）。
- **刚度签名**：他们 dF/dz=−1204（弹簧式，下沉被抵抗）；我们崩塌前 **+980（反弹簧，载荷离开脚面=倾覆签名）**。
- **单一根因候选**：踝-地枢轴等效刚度/惯量差 → 摆模式参数（频率/阻尼）不同——与审核方"法向枢轴刚度"假设一致，且"我们植物太快"首次在承重域定量（3-5×）。
- 下一步：② 根俯仰脉冲响应对数衰减（摆模式阻尼的单值可修量）+ 踝 qpos 吸收器检查（载荷分解的传递侧判据）。

## box 脚变体 (08-20 深夜)：崩塌与几何无关——枢轴阻尼确证

- 我们 R3 阶跃协议 box 脚重跑（ISAAC_FEET=0）：**逐位相同崩塌**（1.3-1.4s，z 0.45→0.21，与胶囊扇一致）。box 前缘 +11.5cm vs 胶囊扇 +8.5cm 的 CoP 行程差未改变崩塌时刻 → **非 CoP 行程限制**。
- 按审核方分叉判据 → 枢轴刚度/阻尼确证。P1 对照钉入：关节级 rise90 135-155ms ≈ 他们 160ms（两边同速）——快/慢分裂只在 free-root+接触约束下出现，异常不住关节/驱动/质量，住在根+接触系统的自由响应。
- **结论链（最终）**：A/B 死亡统计闭环——策略学的是"COM 慢响应"时序，我们快响应使修正落点超前 → ori 漂移死亡（ori 11-15 主力）。
- 下一步：② 根俯仰脉冲对数量衰减（摆模式阻尼单值）+ 法向静刚度两侧干净对比。

## 冻结目标倾覆时间常数 (08-20 深夜)：我们 1.2s vs 他们 2.3s+ = 单值可修量

- **普适崩塌**：我们 18/18 阶跃（任意关节/幅度/符号）~1.2-1.4s 崩塌（z 0.21 躺平）；他们 18/18 存活（2s 内下沉 4.5cm = 同一发散的前期）。脉冲 0.15 与 0.02 rad/s 同刻掀翻（112°）。
- **同一物理的两边**：冻结目标协议 = 无平衡控制器的倒立摆（mgh≈250 Nm/rad 不稳定刚度 > 踝 kp 28.5），发散不可避免——差异 = **时间常数 ≈ 2×**（我们 1.2s / 他们 2.3-3s，正弦截断时刻与阶跃存活窗口一致）。
- **单值可修量**：倾覆时间常数比。τ = √(I_eff/(mgh − kp_eff)) → 嫌疑 = I_eff（RC 求解器全身惯量耦合——静态质量审计匹配≠动态 I_eff 匹配）或 kp_eff（枢轴柔度）。
- 修复验收判据：我们倾覆阈值/时间常数 ≥ 他们（0.2 rad 阶跃存活 2s+）。
- 下一步：I_eff 动态探针（free-root 下小角度摆频 ω=√(mgh−kp)/I 直接反解 I_eff）+ 枢轴柔度旋钮扫描。
