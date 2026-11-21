# C-reranker: SigLIP2 retrieval + LoRA-tuned Qwen3-VL-Reranker-2B (v4)

Submission ZIPs are too large for git (4.2-4.5 GB). They are on the team Drive under `haikuthon/`; this folder has the code to rebuild them.
LoRA weights: Drive `haikuthon/qwen_lora_v4/` (PEFT adapter `adapter_model.safetensors` + `adapter_config.json`, raw `lora_ep*.pt`, merged bf16 model `merged/`).

## 한국어 요약

**1단계 (검색)**: `google/siglip2-so400m-patch14-384`를 256px로 실행 (위치 임베딩 보간), fp16, 질의 소문자화, max_length 64. 질의마다 16,000장 전체와 코사인 유사도 → 상위 후보.

**2단계 (재정렬)**: `Qwen/Qwen3-VL-Reranker-2B` + LoRA (r16, α32, 언어 모델의 q/k/v/o/gate/up/down, 2에폭). 서버에는 병합 후 int8(128개 단위 스케일)로 저장하고 로드 시 bf16으로 복원.
- 프롬프트 (system·instruct 없음): `<|im_start|>user\n<Query>:{질의}\n<Document>:{256px 이미지}<|im_end|>\n<|im_start|>assistant\n`
- 점수 = 마지막 토큰 hidden · (W_yes − W_no)
- 학습 음성: 같은 패밀리 형제(가중치 2) + SigLIP 상위 8 + 규칙 기반 캡션 변형(색·무늬·재질 교체, 좌우/위아래/앞뒤 반전, 개수 ±1)

**k와 시간 가드**: 질의별로 1단계 점수 차(≤0.05)로 k를 1→6까지 정하고, 모든 질의의 순위 0·1 → 순위 2 → 순위 3 … 순서로 처리. `predict()` 안에서 780초가 지나면 멈추고, 남은 후보는 SigLIP 순서 유지.

**결과**: val(1,600장 풀) 0.849, 리더보드 **0.3038** (아래 표). 테스트에서는 1단계 recall(R@5 ≈ .43)과 재정렬 깊이가 병목.

## Leaderboard (public, 8,000 queries × 16,000 images)

| submission | stage 1 | stage 2 | R@1 | R@5 | NDCG@5 | score | time |
|---|---|---|---|---|---|---|---|
| vlm_k3_224_v1 (B) | siglip-base-patch16-224 | zero-shot Qwen3-VL-Reranker-2B, k=3, 224 px | .1872 | .4319 | .3112 | .2492 | 12:59 (guard) |
| CLIP+VLM_v1 (A) | CLIP (+SigLIP?) | VLM, k=5 | .2725 | .4644 | .3826 | .3275 | 6:26 |
| **minprompt = v4 (C)** | SigLIP2-so400m @256 | LoRA reranker, adaptive k≤6, 256 px | .2566 | .4338 | .3511 | .3038 | 13:00 (guard) |

What the test told us:
- SigLIP2-so400m@256 has the same R@5 as siglip-base-224 (.434 vs .432): the bigger stage-1 model bought no test recall, only stage-1 time.
- The two stage-1 models find different family members: their top-5 lists share only 1.9 of 5 images on average.
- R@1/R@5 is .59 for both v4 and the zero-shot k=5 submission (.43 for zero-shot k=3). v4 hit the time guard, so it saw only ~3-4 candidates per query; the LoRA's val accuracy (.93 when the answer is in the top-5) did not fully carry over.
- Reranker resolution on valA/4000 pool (fixed k): 256 px 20.2 ms/pair (k5 .819), 224 px 16.8 ms (k5 .807), 192 px 13.9 ms (k5 .801).
- Leak check: on valA queries whose family has no train member, LoRA accuracy (answer in top-5 → ranked 1st) is .927 (v4) / .943 (v2) vs .543 zero-shot, equal to leaky families → the LoRA generalises to unseen families of the same generator.

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
