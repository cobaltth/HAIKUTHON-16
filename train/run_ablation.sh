#!/usr/bin/env bash
# 누적 ablation: 하나씩 켜면서 바로 앞 단계와 비교한다. 사용: bash train/run_ablation.sh /kaggle/working/haikuthon-train
set -e
DATA=${1:?haikuthon-train 경로}
M=${MODEL:-google/siglip-base-patch16-224}
EP=${EPOCHS:-3}
C="--data $DATA --model $M --epochs $EP"
cd "$(dirname "$0")/.."

python train/train_b.py $C --name r0_zeroshot --epochs 0
python train/train_b.py $C --name r1_base
python train/train_b.py $C --name r2_group    --group-batch
python train/train_b.py $C --name r3_flip     --group-batch --flip-swap-p 0.5
python train/train_b.py $C --name r4_textneg  --group-batch --flip-swap-p 0.5 --n-text-neg 2
python train/train_b.py $C --name r5_crop     --group-batch --flip-swap-p 0.5 --n-text-neg 2 --crop-p 0.5

echo; echo "===== 비교 (앞 단계 대비) ====="
prev=r0_zeroshot
for r in r1_base r2_group r3_flip r4_textneg r5_crop; do
  echo "--- $prev → $r"; python eval/compare.py runs/$prev.json runs/$r.json
  prev=$r
done
