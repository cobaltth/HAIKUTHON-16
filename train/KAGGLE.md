# Kaggle에서 B 트랙 ablation 돌리기

규정: 학습은 **Colab·Kaggle 무료 GPU만**. Kaggle 권장 (세션 최대 12시간, 덜 끊김).

## 0. 노트북 설정
- 오른쪽 패널 Settings → **Accelerator: GPU T4** (x2여도 1장만 씀), **Internet: On**

## 1. 코드 받기
```python
!git clone -q -b B-embedding https://github.com/cobaltth/HAIKUTHON-16 /kaggle/working/repo
%cd /kaggle/working/repo
```

## 2. 패키지 (채점 서버와 같은 버전으로 맞춤)
```python
!pip -q install "transformers==5.22.2" "sentencepiece==0.2.3" "huggingface_hub>=1.0"
```
스크립트는 `!python`으로 따로 실행되므로 재시작 없이 바로 다음 셀로 넘어가도 됨.

## 3. 데이터 받기 (약 3.6GB, 몇 분)
```python
!hf download KimWooJin/26-haikuthon-train --repo-type dataset --include "images/*.png" --include "prompts.jsonl" --local-dir /kaggle/working/haikuthon-train
```

## 4. 먼저 한 번만 빠르게 확인 (사전학습 그대로 채점, 1~3분)
```python
!python train/train_b.py --data /kaggle/working/haikuthon-train --name r0_zeroshot --epochs 0
```
→ 점수, 정답 순위 분포 표, 테스트 규모 환산 추론 시간이 나오면 정상.

## 5. 전체 ablation (누적, 실험당 대략 5~10분 예상)
```python
!bash train/run_ablation.sh /kaggle/working/haikuthon-train
```
하나씩 돌리고 싶으면 `train/train_b.py` 를 직접 실행 (옵션은 파일 맨 위 설명 참고).

| 실험 | 추가되는 것 |
|---|---|
| r0_zeroshot | 학습 없음 (사전학습 그대로) |
| r1_base | 기본 FT: 헤드만, 무작위 배치, 3에포크 (스타터킷과 같은 설정) |
| r2_group | + 형제 같은 배치 |
| r3_flip | + 좌우 반전·left↔right 교환 (p=0.5) |
| r4_textneg | + 규칙 기반 가짜 캡션 2개 |
| r5_crop | + 약한 크롭 (0.9~1.0, p=0.5) |

마지막에 앞 단계 대비 `compare.py` 판정이 자동 출력됨. **채택(구간 하한>0)인 것만 다음에 남긴다**는 기준으로 해석.

## 6. 결과 챙기기
- `runs/*.json` 이 실험 결과 (query별 순위 포함). 노트북 Output에서 내려받거나, 노트북 마지막에 출력 내용을 복사해서 공유
- 모델 가중치가 필요하면 해당 실험에 `--save-model` 추가 → `out/<name>/submission/` (model.py + models/, 제출 zip 구조 그대로. zip 만들 때 `__pycache__` 제외)

## 다음 단계 후보 (ablation 이후)
- 학습 층 넓히기: `--trainable last2` (헤드만 학습은 색-물체 결합을 배우기 어려울 수 있음)
- 에포크/학습률: `--epochs 5 --lr 2e-5`
- 최종 제출용: `--train-on all --save-model` (이때 valA 점수는 무의미)
