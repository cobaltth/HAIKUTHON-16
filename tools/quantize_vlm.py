"""제출 폴더의 VLM 가중치를 int8(행 단위 대칭 양자화)로 압축해 새 제출 폴더 + zip을 만든다.

예) python tools/quantize_vlm.py --src /kaggle/working/sub_vlm_k3_224 --out /kaggle/working/sub_vlm_k3_224_q8

- 2차원이고 원소가 100만 개 이상인 가중치(선형층·임베딩)만 int8 + 행별 배율(fp32)로 저장, 나머지는 원래 dtype 유지
- 모델 내부 이름(state_dict) 기준으로 저장하므로 같은 transformers 버전(5.22.2)에서 그대로 복원된다
- 묶인 가중치(tied, lm_head = embed_tokens)는 한 번만 저장
- 복원(int8 → bf16/fp16)은 model.py의 load_model에서 수행 → 추론 시간에 포함되지 않음
"""
import argparse, json, shutil, time, zipfile
from pathlib import Path

import torch
from safetensors.torch import save_file

ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True, help="기존 제출 폴더 (models/vlm 포함)")
ap.add_argument("--out", required=True)
ap.add_argument("--min-numel", type=int, default=1_000_000)
ap.add_argument("--no-zip", action="store_true")
a = ap.parse_args()

from transformers import Qwen3VLForConditionalGeneration

src, out = Path(a.src), Path(a.out)
if out.exists():
    shutil.rmtree(out)
def _ign(d, names):
    # VLM 원본 가중치만 제외하고 SigLIP 등 나머지 가중치는 그대로 복사
    skip = {n for n in names if n in (".cache", "__pycache__")}
    if Path(d).resolve() == (src / "models" / "vlm").resolve():
        skip |= {n for n in names if n.endswith(".safetensors") or n.endswith(".safetensors.index.json")}
    return skip
shutil.copytree(src, out, ignore=_ign)
vdir = out / "models/vlm"

t = time.time()
lm = Qwen3VLForConditionalGeneration.from_pretrained(src / "models/vlm", dtype=torch.bfloat16)
sd = lm.state_dict()
seen, tensors, meta = {}, {}, {"quantized": [], "kept": [], "tied": {}}
for name, w in sd.items():
    ptr = w.data_ptr()
    if ptr in seen and w.numel() > 0:          # 묶인 가중치는 원본 이름만 저장
        meta["tied"][name] = seen[ptr]
        continue
    seen[ptr] = name
    if w.dim() == 2 and w.numel() >= a.min_numel and w.is_floating_point() and "pos_embed" not in name:
        wf = w.float()
        scale = wf.abs().amax(dim=1).clamp(min=1e-8) / 127.0
        q = torch.round(wf / scale[:, None]).clamp(-127, 127).to(torch.int8)
        tensors[name] = q.contiguous()
        tensors[name + ".__scale"] = scale.contiguous()
        meta["quantized"].append(name)
    else:
        tensors[name] = w.contiguous()
        meta["kept"].append(name)
save_file(tensors, vdir / "model_q8.safetensors", metadata={"format": "pt", "q8": "rowwise-symmetric"})
json.dump(meta, open(vdir / "q8_meta.json", "w"))
print(f"양자화: {len(meta['quantized'])}개 int8, 유지 {len(meta['kept'])}개, 묶음 {len(meta['tied'])}개 ({time.time()-t:.0f}s)")

# 재구성 오차 점검 (임의 몇 개)
err = []
for name in meta["quantized"][:5] + meta["quantized"][-3:]:
    deq = tensors[name].float() * tensors[name + ".__scale"][:, None]
    err.append((name, float((deq - sd[name].float()).abs().mean() / sd[name].float().abs().mean())))
for n, e in err:
    print(f"   상대오차 {e:.4f}  {n}")
del lm, sd, tensors

files = [p for p in out.rglob("*") if p.is_file() and "__pycache__" not in p.parts]
total = sum(p.stat().st_size for p in files)
print(f"새 제출 폴더 {out}: {total/1e9:.2f} GB")
if not a.no_zip:
    zp = out.with_suffix(".zip")
    with zipfile.ZipFile(zp, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as z:
        for p in sorted(files):
            z.write(p, arcname=p.relative_to(out))
    print(f"zip: {zp} ({zp.stat().st_size/1e9:.2f} GB / {zp.stat().st_size} bytes)")
