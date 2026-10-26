"""Write LingBot-VLA 2.0's RoboCasa365 post-training config (configs/vla/robocasa365/robocasa365.yaml) from the repo's own
RoboTwin recipe: our checkpoints and dataset, the full pretraining joint layout (arm, end, effector, waist, head, base,
hand; slots this robot does not use stay padded), AdamW instead of Muon (the README's faster default), and the batch
size / steps for our GPUs.  Usage: python 135_lingbot_cfg.py NGPU MICRO_BATCH MAX_STEPS SAVE_STEPS OUTPUT_DIR
"""
import os
import sys

import yaml

L = "/export/home/kaziz/g05/baselines/lingbot-vla-v2"
M = "/export/home/kaziz/g05/models/lingbot-vla-v2-6b"
ngpu, mb, steps, save, out = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
c = yaml.safe_load(open(f"{L}/configs/vla/robotwin/robotwin.yaml"))
c["model"].update(model_path=M, tokenizer_path="/export/home/kaziz/g05/models/Qwen3-VL-4B-Instruct")
joints = [("arm.position", 14), ("end.position", 14), ("effector.position", 2), ("waist.position", 4), ("head.position", 2),
          ("base.position", 3), ("hand.position", 12)]
c["data"].update(data_name="robocasa365", train_path="/export/home/kaziz/g05/data/rc365_v30/target50",
                 joints=[{k: v} for k, v in joints], norm_type=[{k: "bounds_99_woclip"} for k, _ in joints])
t = c["train"]
t.pop("optimizer", None)  # AdamW
t.update(output_dir=out, micro_batch_size=mb, global_batch_size=mb * ngpu, max_steps=steps, save_steps=save)
# 6B parameters with AdamW need ~16 bytes/parameter: shard everything across our few GPUs (the recipe assumes 32) and
# recompute activations (GC=0 turns it off)
t.update(enable_full_shard=True, enable_gradient_checkpointing=os.environ.get("GC", "1") == "1")
# one GPU: OPT=anyprecision_adamw MIXED=0 keeps weights and optimizer state in bf16 (LingBot's AnyPrecisionAdamW,
# Kahan-compensated unless LB_KAHAN=0), ~8-10 bytes/parameter instead of 16
if os.environ.get("OPT"):
    t["optimizer"] = os.environ["OPT"]
if os.environ.get("MIXED") == "0":  # bf16 weights: 135 MODE=auto1 then runs the forward pass under bf16 autocast (LB_AUTOCAST)
    t["enable_mixed_precision"] = False
if os.environ.get("COMPILE") in ("0", "1"):  # torch.compile on/off (the recipe compiles)
    t["use_compile"] = os.environ["COMPILE"] == "1"
a = t["align_params"]
a["depth"].update(moge_path="/export/home/kaziz/g05/models/moge-2-vitb-normal/model.pt", morgbd_path=f"{M}/depth/model.pt")
a["video"].update(ckpt_path=f"{M}/dino_video/teacher_step_10000.pth", config_path=f"{M}/dino_video/config.yaml")
os.makedirs(f"{L}/configs/vla/robocasa365", exist_ok=True)
yaml.safe_dump(c, open(f"{L}/configs/vla/robocasa365/robocasa365.yaml", "w"), sort_keys=False)
print("wrote configs/vla/robocasa365/robocasa365.yaml:", {k: t[k] for k in ("micro_batch_size", "global_batch_size", "max_steps", "save_steps")})
