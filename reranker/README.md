# C-reranker: SigLIP2 retrieval + LoRA-tuned Qwen3-VL-Reranker-2B (v4)

Submission ZIPs are too large for git (4.2-4.5 GB). They are on the team Drive under `haikuthon/`; this folder has the code to rebuild them.

## Pipeline (`model.py`, the submission entry point)

```
query text ─┐
            ├─ stage 1: SigLIP2-so400m-patch14-384 run at 256 px (interpolated pos-emb), fp16
images ─────┘           scores all candidates -> top-k per query
            └─ stage 2: Qwen3-VL-Reranker-2B + LoRA (merged, stored int8 with per-128 scales, restored to bf16 at load)
                        prompt "<Query>: ... <Document>: [256 px image]" (no system / instruct), score = h_last·(W_yes − W_no)
                        reorders the top-k -> top-5
```

- **Per-query k** (uses only that query's own stage-1 scores): start at `k_min=1`, add candidate r while `s[0] − s[r] ≤ gap` (0.05), up to `k_max=6`.
- **Work order + time guard**: ranks 0–1 of every reranked query first, then rank 2 of all, then rank 3 …; reranking stops at 780 s inside `predict()`. Queries not reached keep the SigLIP order, so a slow server only loses the deepest candidates.
- **Decode once**: each image is decoded once into the SigLIP input and a 256 px copy for the reranker.
- **Query independence**: no cross-query normalisation, dedup or matching.

## Results (team split, valA queries; submission code path, int8)

| version | change | valA (1600 pool) | 4000 pool* | rerank ms/pair (L4) |
|---|---|---|---|---|
| SigLIP2 only | | 0.518 | 0.432 | – |
| v1 | + zero-shot VLM rerank (binding instruct), adaptive k 1→5 | 0.660 | 0.567 | 26.4 |
| v2 | + reranker LoRA (train split, 2 epochs) | 0.810 | 0.744 | 26.4 |
| v3 | + SigLIP at 256 px (stage 1 ~2.5 min / 16k imgs), k 1→8 gap .03 | 0.844 | – | 26.6 |
| **v4** | + prompt removed (LoRA retrained) | **0.849** (unconstrained) | ~0.75–0.78 at 3.5–4 pairs/q | **20.2** |

*4000 pool = valA queries × all 4000 images; LoRA models saw the 2400 train images in training, so slightly optimistic.
Leaderboard has 16 000 candidates per query; local scores drop with pool size (SigLIP: 0.518 → 0.432 → LB 0.23).

Fixed-k ceiling with the LoRA reranker (valA 1600): k=2 .69, 3 .79, 4 .85, 5 .89, 6 .91, 8 .95 → after LoRA, **pairs per query ≈ score**; speed work (256 px stage 1, no prompt) buys depth.

## Rebuild

```bash
# 0) stage-1 weights: models/siglip2 (fp16 SigLIP2 + tokenizer + haiku_preprocess.json), e.g. from prepare_submission.py
# 1) reranker LoRA (needs a folder with model.py + models/siglip2 for mining/eval): ~25 min on L4
python train_lora.py --sub <sub_dir> --out lora_min --instruct "" --system "" --epochs 2
# 2) pack: int8 reranker + SigLIP + model.py + v1_config.json, self-test, ZIP (<5 GB)
python prepare_v1.py --siglip_dir <sub_dir>/models/siglip2 --reranker lora_min/merged --out sub_v4 \
    --check_images haikuthon-train/images --config "$(cat configs/v4_config.json)"
# 3) validate like the leaderboard (add --pool_all for 4000 candidates, --which valB for the lockbox)
python eval_v1.py --sub sub_v4 --which valA
# 4) k rule: dump fixed k=8, simulate rules under time-guard budgets
python eval_v1.py --sub sub_v4 --which valA --no_stage1_only --config '{"k_min":8,"k_max":8}' --dump runs/k8.npz
python sim_k.py runs/k8.npz --ms 20.2 --budget_min 10
```

## Files
- `model.py` — submission (`load_model`, `predict`); defaults = v4 (`configs/v4_config.json`)
- `prepare_v1.py` — int8 packing (+ bf16/int8 agreement check), self-test, ZIP
- `eval_v1.py` — leaderboard-style score (½(R@1+NDCG@5)), timing, server-time estimate, dumps
- `train_lora.py` — reranker LoRA: positive vs sibling / SigLIP-hard negatives + rule-based caption negatives
- `sim_k.py` — offline k-rule search incl. pass-ordered time-guard simulation, answer-rank tables
- `stage1_bench.py` — SigLIP resolution/checkpoint speed vs recall
- `siglip_ft.py` — stage-1 LoRA fine-tuning experiment (sibling batches + caption swaps)

## Rejected (with numbers in the project notes)
caption pipeline (≤ VLM on every query type, +2 GB), sentence-split scoring, long instruct (23% slower for +0.02), ensembles (eat the time budget), cross-query tricks (rule violation).
