#!/usr/bin/env python
"""Build the v1 submission folder + ZIP (run on Colab/L4).

  python prepare_v1.py --siglip_dir sub_siglip2/models/siglip2 --out sub_v1 --check_images haikuthon-train/images

Steps: copy SigLIP2 -> pack Qwen3-VL-Reranker-2B as int8 (per-row scales) -> check int8 vs bf16 scores on real pairs
-> copy model.py -> self-test load_model/predict -> ZIP (model.py at root, must stay under 5GB).
"""
import argparse
import glob
import importlib.util
import json
import os
import shutil
import sys
import zipfile

import torch
from PIL import Image
from safetensors.torch import save_file
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

HERE = os.path.dirname(os.path.abspath(__file__))


def pack_int8(lm, out_path, min_numel=1_000_000, group=0):
    """int8 weight-only. group=0: one scale per row; group=g: one scale per g consecutive inputs (more accurate)."""
    sd = lm.state_dict()
    tied = getattr(lm.config, "tie_word_embeddings", False) or getattr(getattr(lm.config, "text_config", None), "tie_word_embeddings", False)
    out, nq, seen = {}, 0, set()
    for k, v in sd.items():
        if tied and k.startswith("lm_head."):
            continue
        if v.data_ptr() in seen and v.numel() > 0:
            continue
        seen.add(v.data_ptr())
        v = v.detach().cpu()
        if v.is_floating_point() and v.dim() == 2 and v.numel() >= min_numel:
            vf = v.float()
            g = group if group and vf.shape[1] % group == 0 else vf.shape[1]
            vg = vf.view(vf.shape[0], -1, g)
            scale = vg.abs().amax(dim=2).clamp(min=1e-8) / 127.0          # (out, in/g)
            out[k] = torch.round(vg / scale[:, :, None]).clamp(-127, 127).to(torch.int8).view_as(vf).contiguous()
            out[k + ".__scale"] = scale.to(torch.float16).contiguous()
            nq += 1
        else:
            out[k] = v.to(torch.bfloat16).contiguous() if v.is_floating_point() else v.contiguous()
    save_file(out, out_path)
    print(f"[pack] {nq} tensors int8, tied_lm_head_dropped={tied}, file {os.path.getsize(out_path) / 1e9:.2f} GB")


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, os.path.dirname(path))
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--siglip_dir", default="sub_siglip2/models/siglip2")
    ap.add_argument("--reranker", default="Qwen/Qwen3-VL-Reranker-2B")
    ap.add_argument("--out", default="sub_v1")
    ap.add_argument("--config", default="{}", help='JSON overrides written to models/v1_config.json, e.g. {"k_min":3}')
    ap.add_argument("--check_images", default="haikuthon-train/images")
    ap.add_argument("--skip_pack", action="store_true")
    ap.add_argument("--group", type=int, default=128, help="int8 scale group size (0 = per row)")
    a = ap.parse_args()

    md = os.path.join(a.out, "models")
    os.makedirs(md, exist_ok=True)
    if not os.path.exists(os.path.join(md, "siglip2")):
        shutil.copytree(a.siglip_dir, os.path.join(md, "siglip2"))
    rdir = os.path.join(md, "reranker")
    os.makedirs(rdir, exist_ok=True)
    shutil.copy(os.path.join(HERE, "model.py"), os.path.join(a.out, "model.py"))
    with open(os.path.join(md, "v1_config.json"), "w") as f:
        json.dump(json.loads(a.config), f)

    if not a.skip_pack:
        lm = Qwen3VLForConditionalGeneration.from_pretrained(a.reranker, dtype=torch.bfloat16)
        lm.config.save_pretrained(rdir)
        AutoProcessor.from_pretrained(a.reranker).save_pretrained(rdir)
        pack_int8(lm, os.path.join(rdir, "int8.safetensors"), group=a.group)

        # int8 vs bf16 check on real pairs
        mod = load_module(os.path.join(a.out, "model.py"), "v1_model_check")
        lm.to("cuda").eval()
        proc = AutoProcessor.from_pretrained(a.reranker); proc.tokenizer.padding_side = "left"
        voc = proc.tokenizer.get_vocab(); W = lm.get_output_embeddings().weight
        w_ref = (W[voc["yes"]].float() - W[voc["no"]].float())
        lm8, proc8, w8 = mod._load_reranker(rdir, "cuda")
        imgs = sorted(glob.glob(os.path.join(a.check_images, "*.png")))[:24]
        ims = [Image.open(p).convert("RGB").resize((256, 256)) for p in imgs]
        q = "One blue screwdriver is to the left of one polka-dotted mug that rests on one stool."
        fake = {"cfg": dict(mod.CFG), "proc": proc}
        txt = [mod._prompt(fake, q)] * len(ims)
        with torch.inference_mode():
            inp = proc(text=txt, images=ims, padding=True, return_tensors="pt", do_resize=False).to("cuda")
            s_ref = (lm.model(**inp).last_hidden_state[:, -1].float() @ w_ref).cpu()
            s_8 = (lm8.model(**inp).last_hidden_state[:, -1].float() @ w8).cpu()
        same = int((s_ref.argsort(descending=True)[:5] == s_8.argsort(descending=True)[:5]).sum())
        print(f"[check] bf16 vs int8 on {len(ims)} images: max |diff| {float((s_ref - s_8).abs().max()):.4f}, "
              f"corr {float(torch.corrcoef(torch.stack([s_ref, s_8]))[0, 1]):.4f}, top-5 order identical {same}/5")
        del lm, lm8
        torch.cuda.empty_cache()

    # self-test of the submission interface
    mod = load_module(os.path.join(a.out, "model.py"), "v1_model_selftest")
    m = mod.load_model(os.path.abspath(a.out), "cuda:0")
    imgs = sorted(glob.glob(os.path.join(a.check_images, "*.png")))[:40]
    cands = [{"image_id": os.path.basename(p)[:-4], "image_path": os.path.abspath(p)} for p in imgs]
    queries = [{"prompt_id": f"q{i}", "prompt": f"test query number {i} with one red object."} for i in range(6)]
    preds = mod.predict(m, queries, cands, top_k=5)
    ids = {c["image_id"] for c in cands}
    for q in queries:
        p = preds[q["prompt_id"]]
        assert len(p) == 5 and len(set(p)) == 5 and all(x in ids for x in p), p
    print("[selftest] output format OK")
    del m
    torch.cuda.empty_cache()

    zp = a.out.rstrip("/") + ".zip"
    with zipfile.ZipFile(zp, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as z:
        for root, dirs, files in os.walk(a.out):
            dirs[:] = [d for d in dirs if d != "__pycache__"]
            for fn in files:
                full = os.path.join(root, fn)
                z.write(full, os.path.relpath(full, a.out))
    size = os.path.getsize(zp) / 1024 ** 3
    print(f"[zip] {zp}: {size:.2f} GB {'OK' if size < 5 else 'TOO BIG (>5GB)'}")


if __name__ == "__main__":
    main()
