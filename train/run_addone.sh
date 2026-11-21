#!/usr/bin/env bash
# 개별(add-one) ablation: 기준 a1 = last2 기본 FT, 나머지는 a1 + 증강 하나씩
set -e
nvidia-smi --query-gpu=name,memory.total --format=csv
cd /kaggle/working
[ -d repo ] || git clone -q -b B-embedding https://github.com/cobaltth/HAIKUTHON-16 repo
pip -q install "transformers==5.22.2" "sentencepiece==0.2.3" "huggingface_hub>=1.0" 2>&1 | tail -2
[ -f haikuthon-train/prompts.jsonl ] && [ $(ls haikuthon-train/images | wc -l) -eq 4000 ] || hf download KimWooJin/26-haikuthon-train --repo-type dataset --include "images/*.png" --include "prompts.jsonl" --local-dir /kaggle/working/haikuthon-train > /kaggle/working/dl.log 2>&1
ls haikuthon-train/images | wc -l
D=/kaggle/working/haikuthon-train
C="--data $D --model google/siglip-base-patch16-224 --epochs 3 --trainable last2"
cd /kaggle/working/repo
python train/train_b.py --data $D --model google/siglip-base-patch16-224 --name r0_zeroshot --epochs 0
python train/train_b.py $C --name a1_last2
python train/train_b.py $C --name a2_group   --group-batch
python train/train_b.py $C --name a3_flip    --flip-swap-p 0.5
python train/train_b.py $C --name a4_textneg --n-text-neg 2
python train/train_b.py $C --name a5_crop    --crop-p 0.5
echo "===== r0 → a1 (last2 기본 FT)"; python eval/compare.py runs/r0_zeroshot.json runs/a1_last2.json
for r in a2_group a3_flip a4_textneg a5_crop; do echo "===== a1 → $r"; python eval/compare.py runs/a1_last2.json runs/$r.json; done
echo ALLDONE
