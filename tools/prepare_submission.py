"""제출 폴더 + zip 만들기 (Kaggle/Colab에서 실행, 인터넷 필요).

예) SigLIP + Qwen3-VL-Reranker 재정렬 (K=5, VLM 이미지 448px)
    python tools/prepare_submission.py --out /kaggle/working/sub_vlm_k5 --k 5 --vlm-size 448
    SigLIP만:   --no-rerank
    B 트랙 학습 결과를 1단계로: --siglip out/<실험이름>/submission/models

만들어지는 것
    <out>/model.py, requirements.txt, models/siglip/(fp16), models/vlm/, models/config_b.json
    <out>.zip  (model.py가 zip 최상위)
마지막에 작은 샘플로 load_model → predict(top_k=5) 형식 검사까지 한다.
"""
import argparse, importlib.util, json, shutil, sys, time, zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval"))

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--out", required=True)
ap.add_argument("--siglip", default="google/siglip-base-patch16-224")
ap.add_argument("--vlm", default="Qwen/Qwen3-VL-Reranker-2B")
ap.add_argument("--no-rerank", action="store_true")
ap.add_argument("--k", type=int, default=5)
ap.add_argument("--vlm-size", type=int, default=448)
ap.add_argument("--vlm-batch", type=int, default=20)
ap.add_argument("--time-budget", type=int, default=780, help="predict 전체 시간 예산(초), 서버 제한 900초")
ap.add_argument("--margin-threshold", type=float, default=None)
ap.add_argument("--fuse-alpha", type=float, default=0.0)
ap.add_argument("--data", default="/kaggle/working/haikuthon-train", help="형식 자체 테스트용 데이터")
ap.add_argument("--no-zip", action="store_true")
ap.add_argument("--no-selftest", action="store_true")
a = ap.parse_args()

import torch
from transformers import AutoProcessor, SiglipModel

out = Path(a.out)
if out.exists():
    shutil.rmtree(out)
(out / "models").mkdir(parents=True)

# 1) SigLIP (fp16으로 저장 → 용량 절반)
t = time.time()
sig = SiglipModel.from_pretrained(a.siglip, dtype=torch.float16)
sig.save_pretrained(out / "models/siglip")
AutoProcessor.from_pretrained(a.siglip).save_pretrained(out / "models/siglip")
print(f"SigLIP 저장 ({time.time()-t:.0f}s)")

# 2) VLM
if not a.no_rerank:
    from huggingface_hub import snapshot_download
    t = time.time()
    snapshot_download(a.vlm, local_dir=out / "models/vlm",
                      allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja", "*.model"],
                      ignore_patterns=["1_*/*", "scripts/*", "additional_chat_templates/*"])
    print(f"VLM 저장 ({time.time()-t:.0f}s)")

# 3) 설정 + 코드
cfg = {"rerank": not a.no_rerank, "k": a.k, "vlm_size": a.vlm_size, "vlm_batch": a.vlm_batch,
       "time_budget_sec": a.time_budget, "margin_threshold": a.margin_threshold, "fuse_alpha": a.fuse_alpha,
       "source": {"siglip": a.siglip, "vlm": None if a.no_rerank else a.vlm}}
json.dump(cfg, open(out / "models/config_b.json", "w"), indent=1)
shutil.copy(ROOT / "submission_vlm/model.py", out / "model.py")
shutil.copy(ROOT / "submission_template/requirements.txt", out / "requirements.txt")

# 4) 용량
files = [p for p in out.rglob("*") if p.is_file() and "__pycache__" not in p.parts]
total = sum(p.stat().st_size for p in files)
print(f"총 용량 {total/1e9:.2f} GB (제한 5GB)")
for p in sorted(files, key=lambda p: -p.stat().st_size)[:5]:
    print(f"   {p.stat().st_size/1e9:6.2f} GB  {p.relative_to(out)}")
if total > 4.95e9:
    print("⚠ 5GB 제한에 너무 가깝거나 초과! (VLM 양자화나 더 작은 SigLIP 검토)")

# 5) 형식 자체 테스트 (작은 샘플)
if not a.no_selftest and Path(a.data).exists():
    from harness import check_predictions
    rows = [json.loads(l) for l in open(Path(a.data) / "prompts.jsonl", encoding="utf-8") if l.strip()][:40]
    queries = [{"prompt_id": r["id"], "prompt": r["prompts"]} for r in rows[:20]]
    cands = [{"image_id": r["id"], "image_path": str((Path(a.data) / "images" / f"{r['id']}.png").resolve())} for r in rows]
    spec = importlib.util.spec_from_file_location("sub_model", out / "model.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = mod.load_model(str(out), dev)
    preds = mod.predict(model, queries, cands, top_k=5)
    errs = check_predictions(preds, queries, cands, 5)
    hit = sum(preds[q["prompt_id"]][0] == q["prompt_id"] for q in queries)
    print("형식 검사:", "통과" if not errs else "실패 " + " / ".join(errs), f"| 샘플 20개 중 1등 적중 {hit}")
    del model; torch.cuda.empty_cache() if torch.cuda.is_available() else None

# 6) zip (압축 안 함: 가중치는 압축 효과가 거의 없고 시간만 듦)
if not a.no_zip:
    t = time.time()
    zpath = out.with_suffix(".zip")
    with zipfile.ZipFile(zpath, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as z:
        for p in sorted(files):
            z.write(p, arcname=p.relative_to(out))
    print(f"zip: {zpath} ({zpath.stat().st_size/1e9:.2f} GB, {time.time()-t:.0f}s)")
