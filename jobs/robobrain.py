"""RoboBrain 2.0 / 2.5 (BAAI) as a high-level module: multi-image questions -> short text answers.

Follows the reference inference code (AutoModelForImageTextToText, plain user prompt, greedy decoding). Runs on its own
GPU next to the G0.5 server (fp16 on T4/V100, bf16 on Ampere+). Batched: several (images, question) pairs per generate().
"""
import re

import numpy as np
import torch
from PIL import Image


class RoboBrain:
    def __init__(self, path, device="cuda:1", max_side=256):
        from transformers import AutoModelForImageTextToText, AutoProcessor
        self.device = torch.device(device)
        cap = torch.cuda.get_device_capability(self.device)[0]
        self.dtype = torch.bfloat16 if cap >= 8 else torch.float16
        self.model = AutoModelForImageTextToText.from_pretrained(path, torch_dtype=self.dtype, device_map={"": str(self.device)},
                                                                 attn_implementation="sdpa").eval()
        self.proc = AutoProcessor.from_pretrained(path)
        self.proc.tokenizer.padding_side = "left"
        self.max_side = max_side
        self.n_calls = 0

    def _pil(self, im):
        im = Image.fromarray(np.asarray(im, dtype=np.uint8)) if not isinstance(im, Image.Image) else im
        if max(im.size) > self.max_side:
            s = self.max_side / max(im.size)
            im = im.resize((round(im.width * s), round(im.height * s)))
        return im

    @torch.no_grad()
    def ask(self, batch, max_new_tokens=12):
        """batch: list of (list_of_images, question). Returns list of answer strings."""
        texts, images = [], []
        for ims, q in batch:
            pil = [self._pil(x) for x in ims]
            msg = [{"role": "user", "content": [{"type": "image", "image": p} for p in pil] + [{"type": "text", "text": q}]}]
            texts.append(self.proc.apply_chat_template(msg, tokenize=False, add_generation_prompt=True))
            images.append(pil)
        inputs = self.proc(text=texts, images=images, padding=True, return_tensors="pt").to(self.device)
        out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        self.n_calls += len(batch)
        return self.proc.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True, clean_up_tokenization_spaces=False)


def first_int(text, lo=None, hi=None):
    """First integer in the answer (also handles 'zero'...'ten'); None if absent or out of range."""
    words = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}
    m = re.search(r"-?\d+", text)
    v = int(m.group()) if m else next((n for w, n in words.items() if re.search(rf"\b{w}\b", text.lower())), None)
    if v is None or (lo is not None and v < lo) or (hi is not None and v > hi):
        return None
    return v


def task_choice_prompt(commands, n_images, mode):
    opts = "\n".join(f"{k + 1}. {c}" for k, c in enumerate(commands))
    if mode == "current":
        head = "The image shows a robot arm in a kitchen, in the middle of performing a task."
    else:
        head = f"The {n_images} images show a robot arm in a kitchen over the last seconds, in time order (the last image is now)."
    return f"{head}\nWhich of these tasks is the robot performing?\n{opts}\nAnswer with the number of the task only."


def count_prompt(obj, n_images, mode):
    what = "lifting the bowl and placing it back on the plate" if obj == "bowl" else "lifting the bottle and putting it down on the plate"
    if mode == "current":
        return (f"The image shows a robot arm repeatedly {what}. How many times has the robot completed this so far? "
                "Answer with a single number.")
    return (f"The {n_images} images show a robot arm over the whole episode so far, in time order (oldest first, the last image is now). "
            f"The robot is repeatedly {what}. How many times has it completed this so far? Answer with a single number.")


def goal_choice_prompt(commands):
    opts = "\n".join(f"{k + 1}. {c}" for k, c in enumerate(commands))
    return ("Image 1 is the current scene. Image 2 shows the goal: how the scene should look when the task is done.\n"
            f"Which instruction should the robot execute to reach the goal?\n{opts}\nAnswer with the number of the instruction only.")
