"""High-level planner server: Qwen3-VL-4B-Instruct (Apache-2.0) answering text questions about camera images, for policy
wrappers that run in other environments (planned_rc365_policy.py). Protocol: multiprocessing.connection, one request =
{"images": [HxWx3 uint8, ...], "prompt": str, "max_new_tokens": int} -> {"text": str, "seconds": float}.
Run with the Awomo venv (its transformers loads Qwen3-VL): python 128_planner_server.py --port P [--model DIR]
"""
import argparse
import time
from multiprocessing.connection import Listener

import numpy as np
import torch
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, required=True)
ap.add_argument("--model", default="/export/home/kaziz/g05/models/Qwen3-VL-4B-Instruct")
a = ap.parse_args()

from transformers import AutoProcessor, Qwen3VLForConditionalGeneration  # noqa: E402

dtype = torch.float16 if torch.cuda.get_device_capability()[0] < 8 else torch.bfloat16
model = Qwen3VLForConditionalGeneration.from_pretrained(a.model, torch_dtype=dtype, attn_implementation="sdpa").to("cuda").eval()
proc = AutoProcessor.from_pretrained(a.model)
print(f"[planner] {a.model} loaded in {dtype}; listening on {a.port}", flush=True)


def answer(images, prompt, max_new_tokens):
    content = [{"type": "image", "image": Image.fromarray(np.asarray(im, np.uint8))} for im in images] + [{"type": "text", "text": prompt}]
    msgs = [{"role": "user", "content": content}]
    inputs = proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return proc.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()


with Listener(("127.0.0.1", a.port), authkey=b"g05-planner") as lst:
    while True:
        conn = lst.accept()
        try:
            while True:
                req = conn.recv()
                t0 = time.perf_counter()
                txt = answer(req["images"], req["prompt"], int(req.get("max_new_tokens", 64)))
                conn.send({"text": txt, "seconds": time.perf_counter() - t0})
        except (EOFError, ConnectionResetError):
            conn.close()
