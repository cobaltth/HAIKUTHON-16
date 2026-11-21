"""LoRA를 합친 BLIP-2 ITM: fp16(autocast) 경로와 fp32 경로의 점수 비교 (검증 40세트, 세트 안 8×8 쌍 2,560개)."""
import importlib.util, json, random, sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image

spec = importlib.util.spec_from_file_location("m5", sys.argv[1]); m5 = importlib.util.module_from_spec(spec); spec.loader.exec_module(m5)
lora = sys.argv[2]
sys.path.insert(0, "/kaggle/working/repo")
from train.blip_ft import scenes_of
dev = "cuda"
caps = {}
for l in open("/kaggle/working/haikuthon-train/prompts.jsonl", encoding="utf-8"):
    if l.strip():
        r = json.loads(l); caps[r["id"]] = r["prompts"]
sc = scenes_of(caps); random.Random(0).shuffle(sc); val = sc[:40]
ids = [i for g in val for i in g]
blip, proc = m5._load_blip(Path("/kaggle/working/sub_ens_k15_t840/models/blip"), dev, lora)
imgs = np.stack([np.asarray(Image.open(f"/kaggle/working/haikuthon-train/images/{i}.png").convert("RGB").resize((224, 224), Image.Resampling.BICUBIC)) for i in ids])
vdt = blip.vision_model.embeddings.patch_embedding.weight.dtype
out = {}
with torch.inference_mode():
    E = torch.cat([m5._blip_image_embeds(blip, m5._to_pixels(imgs[s:s + 48], m5.CLIP_MEAN, m5.CLIP_STD, dev, vdt)) for s in range(0, len(ids), 48)])
    tok = proc.tokenizer([caps[i] for i in ids], padding="longest", truncation=True, max_length=128, return_tensors="pt").to(dev)
    for name, on in (("fp16", True), ("fp32", False)):
        L = torch.empty(40, 8, 8)
        for s in range(40):
            ti = torch.arange(s * 8, s * 8 + 8, device=dev).repeat_interleave(8)
            ii = torch.arange(s * 8, s * 8 + 8, device=dev).repeat(8)
            emb = E[ii] if on else E[ii].float()
            if not on:
                blip.itm_head.float()
            with torch.autocast("cuda", dtype=torch.float16, enabled=on):
                v = m5._blip_itm(blip, emb, tok.input_ids[ti], tok.attention_mask[ti])
            L[s] = v.float().cpu().view(8, 8)
        out[name] = L
d = (out["fp16"] - out["fp32"]).abs()
ar = torch.arange(8)
r1 = {k: round(float((v.argmax(2) == ar).float().mean()), 4) for k, v in out.items()}
agree = float((out["fp16"].argmax(2) == out["fp32"].argmax(2)).float().mean())
gap = out["fp32"].sort(2, descending=True).values
print(f"FPCHK lora={lora} pairs {d.numel()} | max diff {float(d.max()):.4f} mean diff {float(d.mean()):.4f} | logit std {float(out['fp32'].std()):.2f} "
      f"mean top1-top2 gap {float((gap[:, :, 0] - gap[:, :, 1]).mean()):.2f} | top1 agree {agree:.4f} | R@1 {r1}", flush=True)
