#!/usr/bin/env python3
"""Add detailed logging instrumentation to ppo_trainer.py for the training report.

Run this inside the container before starting training:
    python3 scripts/add_training_instrumentation.py

It patches gear_sonic/trl/trainer/ppo_trainer.py to log:
- obs shapes at first iteration
- action distribution stats
- per-loss component values
- gradient norms
- advantage/value stats
- data loading info
"""

import os
import re

ROOT = "/root/GR00T-WholeBodyControl"

# ── Patch 1: Add detailed logging to _rollout_step ──
ROLLOUT_PATCH_MARKER = "self.storage.batch_update_data(\"advantages\", advantages)"
ROLLOUT_PATCH_INSERT = '''
                # ── Instrumentation: log rollout stats ──
                if self.accelerator.is_main_process and (self.state.global_step <= 2 or self.state.global_step % 50 == 0):
                    import logging
                    _logger = logging.getLogger("sonic_instrument")
                    _it = self.state.global_step + 1
                    _logger.info(f"=== ROLLOUT STATS [iter {_it}] ===")
                    for _k, _v in obs_dict.items():
                        _logger.info(f"  obs[{_k}]: shape={_v.shape}, mean={_v.float().mean():.4f}, std={_v.float().std():.4f}")
                    _logger.info(f"  values: shape={values.shape}, mean={values.float().mean():.4f}, std={values.float().std():.4f}")
                    _logger.info(f"  returns: shape={returns.shape}, mean={returns.float().mean():.4f}, std={returns.float().std():.4f}")
                    _logger.info(f"  advantages: shape={advantages.shape}, mean={advantages.float().mean():.4f}, std={advantages.float().std():.4f}")
                    _r = self.storage.query_key("rewards")
                    _logger.info(f"  rewards: shape={_r.shape}, mean={_r.float().mean():.4f}, std={_r.float().std():.4f}, min={_r.float().min():.4f}, max={_r.float().max():.4f}")
                    _logger.info(f"=== END ROLLOUT STATS ===")
'''

# ── Patch 2: Add model architecture logging at init ──
INIT_PATCH_MARKER = "self.log(metrics)"
INIT_PATCH_INSERT = '''
                # ── Instrumentation: log model arch + loss details ──
                if self.accelerator.is_main_process and self.state.global_step == 1:
                    import logging
                    _logger = logging.getLogger("sonic_instrument")
                    _logger.info("=== MODEL ARCHITECTURE ===")
                    _total_params = 0
                    _module_params = {}
                    for _name, _param in self.model.module.named_parameters():
                        _total_params += _param.numel()
                        _top = _name.split(".")[1] if "." in _name else _name
                        _module_params[_top] = _module_params.get(_top, 0) + _param.numel()
                    _logger.info(f"  Total parameters: {_total_params:,}")
                    for _mod, _cnt in sorted(_module_params.items()):
                        _logger.info(f"  {_mod}: {_cnt:,} params")
                    _logger.info("=== END MODEL ARCHITECTURE ===")

                    # Log detailed metrics
                    _logger.info("=== DETAILED TRAINING METRICS [iter 1] ===")
                    for _mk, _mv in sorted(metrics.items()):
                        _logger.info(f"  {_mk}: {_mv}")
                    _logger.info("=== END DETAILED TRAINING METRICS ===")
'''

# ── Patch 3: Add gradient norm logging ──
GRAD_PATCH_MARKER = "self.sync_running_mean_std()"
GRAD_PATCH_INSERT = '''
            # ── Instrumentation: gradient norm ──
            if self.accelerator.is_main_process and (self.state.global_step <= 2 or self.state.global_step % 50 == 0):
                import logging
                _logger = logging.getLogger("sonic_instrument")
                _total_norm = 0.0
                _per_module_norms = {}
                for _name, _param in self.model.module.named_parameters():
                    if _param.grad is not None:
                        _pnorm = _param.grad.data.float().norm(2).item()
                        _total_norm += _pnorm ** 2
                        _top = _name.split(".")[1] if len(_name.split(".")) > 1 else _name
                        _per_module_norms[_top] = _per_module_norms.get(_top, 0.0) + _pnorm ** 2
                _total_norm = _total_norm ** 0.5
                _logger.info(f"=== GRADIENT NORMS [iter {self.state.global_step + 1}] ===")
                _logger.info(f"  total_grad_norm: {_total_norm:.6f}")
                for _mod, _gnorm in sorted(_per_module_norms.items()):
                    _logger.info(f"  {_mod}: {_gnorm ** 0.5:.6f}")
                _logger.info(f"=== END GRADIENT NORMS ===")

'''


def patch_file(filepath, patches):
    """Apply text patches to a file."""
    with open(filepath, "r") as f:
        content = f.read()

    for marker, insert_text in patches:
        if "sonic_instrument" in content and marker in content:
            print(f"  [SKIP] Already patched near '{marker[:50]}...'")
            continue
        if marker not in content:
            print(f"  [WARN] Marker not found: '{marker[:60]}...'")
            continue
        content = content.replace(marker, marker + insert_text, 1)
        print(f"  [OK] Patched near '{marker[:50]}...'")

    with open(filepath, "w") as f:
        f.write(content)


def setup_logging_config():
    """Create a logging setup snippet to prepend to train_agent_trl.py."""
    filepath = os.path.join(ROOT, "gear_sonic", "train_agent_trl.py")
    with open(filepath, "r") as f:
        content = f.read()

    if "sonic_instrument" in content:
        print("[SKIP] train_agent_trl.py already has logging config")
        return

    # Add after imports
    log_setup = '''
# ── Instrumentation logging setup ──
import logging as _logging
_inst_logger = _logging.getLogger("sonic_instrument")
_inst_logger.setLevel(_logging.INFO)
if not _inst_logger.handlers:
    _inst_handler = _logging.StreamHandler()
    _inst_handler.setFormatter(_logging.Formatter("[INSTRUMENT] %(message)s"))
    _inst_logger.addHandler(_inst_handler)
# ── End instrumentation setup ──
'''

    # Find a good insertion point (after the last import)
    lines = content.split("\n")
    insert_idx = 0
    for i, line in enumerate(lines):
        if line.startswith("import ") or line.startswith("from "):
            insert_idx = i + 1

    lines.insert(insert_idx, log_setup)
    with open(filepath, "w") as f:
        f.write("\n".join(lines))
    print(f"[OK] Added logging config to train_agent_trl.py at line {insert_idx}")


def main():
    print("=== Applying training instrumentation ===")

    # 1. Patch ppo_trainer.py
    trainer_path = os.path.join(ROOT, "gear_sonic", "trl", "trainer", "ppo_trainer.py")
    print(f"\nPatching {trainer_path}...")
    patch_file(trainer_path, [
        (ROLLOUT_PATCH_MARKER, ROLLOUT_PATCH_INSERT),
        (INIT_PATCH_MARKER, INIT_PATCH_INSERT),
        (GRAD_PATCH_MARKER, GRAD_PATCH_INSERT),
    ])

    # 2. Setup logging in train_agent_trl.py
    print(f"\nSetting up logging...")
    setup_logging_config()

    print("\n=== Instrumentation complete ===")


if __name__ == "__main__":
    main()
