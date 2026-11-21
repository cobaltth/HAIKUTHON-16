"""앙상블 제출 폴더 만들기 (submission_ens/model.py + 가중치 4종).

예) 학습 없는 대체 모델로 시간·동작 시험:
  python tools/build_ens.py --out /kaggle/working/sub_ens_zs \
     --siglip out/a1_save/submission/models --blip Salesforce/blip2-itm-vit-g \
     --qwen Qwen/Qwen3-VL-Reranker-2B --config '{"qwen_system": "<official>", "qwen_instruct": "<binding>"}'
예) 팀 가중치로:
  python tools/build_ens.py --out sub_ens --siglip <A_zip>/models --blip <A_zip>/reranker \
     --qwen <C_zip>/models/reranker --config '{"blip_weight": <A pipeline.json rerank_weight>}'

--siglip/--blip/--qwen/--siglip2 : 로컬 폴더(그대로 복사) 또는 HF id(받아서 fp16/bf16 저장)
--blip-q8 : BLIP-2를 int8(행 단위)로 저장해 용량 절반
--zip     : 무압축 zip 생성
"""
import argparse, json, shutil, sys, zipfile
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_SYSTEM = ('Judge whether the Document meets the requirements based on the Query and the Instruct provided. '
                   'Note that the answer can only be "yes" or "no".')
BINDING = ("Verify attribute binding and relation direction: every color, count, material, pattern and pose must belong "
           "to the object named in the query, and every spatial relation must point the stated way. "
           "A swapped or reversed detail means no.")


def copy_or_save(src, dst, kind, q8=False, convert=False):
    p = Path(src)
    if p.exists() and not (q8 or convert):
        shutil.copytree(p, dst, ignore=shutil.ignore_patterns(".cache", "__pycache__", "*.zip"))
        return
    from transformers import AutoProcessor
    if kind == "siglip":
        from transformers import SiglipModel as C; dt = torch.float16
    elif kind == "siglip2":
        from transformers import AutoModel as C; dt = torch.float16
    elif kind == "blip":
        from transformers import Blip2ForImageTextRetrieval as C; dt = torch.float16
    else:
        from transformers import Qwen3VLForConditionalGeneration as C; dt = torch.bfloat16
    m = C.from_pretrained(src, dtype=dt)
    dst.mkdir(parents=True)
    if q8:
        quantize_save(m, dst)
        m.config.save_pretrained(dst)
    else:
        m.save_pretrained(dst)
    if kind == "siglip2":
        from transformers import AutoTokenizer
        AutoTokenizer.from_pretrained(src).save_pretrained(dst)
    else:
        AutoProcessor.from_pretrained(src).save_pretrained(dst)


def quantize_save(m, dst, min_numel=1_000_000):
    from safetensors.torch import save_file
    seen, tensors, meta = {}, {}, {"quantized": [], "kept": [], "tied": {}}
    for name, w in m.state_dict().items():
        ptr = w.data_ptr()
        if ptr in seen and w.numel() > 0:
            meta["tied"][name] = seen[ptr]; continue
        seen[ptr] = name
        if w.dim() == 2 and w.numel() >= min_numel and w.is_floating_point() and "pos_embed" not in name:
            wf = w.float(); sc = wf.abs().amax(1).clamp(min=1e-8) / 127.0
            tensors[name] = torch.round(wf / sc[:, None]).clamp(-127, 127).to(torch.int8).contiguous()
            tensors[name + ".__scale"] = sc.contiguous(); meta["quantized"].append(name)
        else:
            tensors[name] = w.contiguous(); meta["kept"].append(name)
    save_file(tensors, dst / "model_q8.safetensors", metadata={"format": "pt"})
    json.dump(meta, open(dst / "q8_meta.json", "w"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--siglip", required=True)
    ap.add_argument("--blip", required=True)
    ap.add_argument("--qwen", default=None)
    ap.add_argument("--siglip2", default=None)
    ap.add_argument("--blip-q8", action="store_true")
    ap.add_argument("--siglip-fp16", action="store_true", help="로컬 폴더 SigLIP도 fp16으로 다시 저장 (용량 절반)")
    ap.add_argument("--config", default="{}")
    ap.add_argument("--zip", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        shutil.rmtree(out)
    (out / "models").mkdir(parents=True)
    shutil.copy(ROOT / "submission_ens" / "model.py", out / "model.py")
    (out / "requirements.txt").write_text("")
    copy_or_save(a.siglip, out / "models/siglip", "siglip", convert=a.siglip_fp16)
    copy_or_save(a.blip, out / "models/blip", "blip", q8=a.blip_q8)
    if a.qwen:
        copy_or_save(a.qwen, out / "models/qwen", "qwen")
    if a.siglip2:
        copy_or_save(a.siglip2, out / "models/siglip2", "siglip2")
    cfg = json.loads(a.config)
    for k in ("qwen_system", "qwen_instruct"):
        if cfg.get(k) == "<official>":
            cfg[k] = OFFICIAL_SYSTEM
        elif cfg.get(k) == "<binding>":
            cfg[k] = BINDING
    if not a.qwen:
        cfg["qwen_weight"] = 0.0
    json.dump(cfg, open(out / "models/ens_config.json", "w"), ensure_ascii=False, indent=1)
    size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
    for d in sorted((out / "models").iterdir()):
        if d.is_dir():
            print(f"  {d.name:10s} {sum(f.stat().st_size for f in d.rglob('*') if f.is_file())/1e9:.2f} GB")
    print(f"폴더 {out}: {size/1e9:.2f} GB | config {cfg}")
    if a.zip:
        zp = out.with_suffix(".zip")
        with zipfile.ZipFile(zp, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as z:
            for f in sorted(out.rglob("*")):
                if f.is_file():
                    z.write(f, f.relative_to(out))
        print(f"zip {zp}: {zp.stat().st_size/1e9:.2f} GB ({zp.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
