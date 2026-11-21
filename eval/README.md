# 로컬 검증 (B 트랙)

- `splits/split_group_seed42.json` : **팀 공용 분할**. 형제 캡션(같은 물체 구성, `eda/groups.csv`)이 같은 세트에 들어가도록 그룹 단위로 나눔. train 2,400 / valA 800(튜닝·비교용) / valB 800(최종 확인용, 자주 보지 말 것)
- `splits/split_random_starterkit.json` : 스타터킷 train.py와 같은 무작위 분할(3,200/800). 비교·가설 검증용
- `eval/metrics.py` : 대회 점수식 (R@1 + NDCG@5)/2, Recall@K, 정답 순위 분포
- `eval/harness.py` : 제출 폴더의 `model.py`를 서버와 같은 방식(load_model → predict)으로 불러 채점

## 사용법 (Colab/Kaggle)
```bash
python eval/harness.py --submission ../starterkit/sample_submission --data ../haikuthon-train --split valA --pool pair --topk 50
```
- `--pool pair` (기본): 후보 = valA+valB 이미지 1,600장 → 테스트와 같은 query:이미지 = 1:2 비율
- `--pool own` : 후보 = 해당 세트 800장 (스타터킷 검증 조건과 동일)
- `--split-file splits/split_random_starterkit.json --pool own` : 스타터킷 수치(0.7839) 재현용
- `--save out.json` : query별 정답 순위·예측 저장 → hard negative 마이닝, 재정렬 K 결정
- `--topk 50` : predict에 top-50까지 받아 Recall@K·순위 분포 계산 (점수는 상위 5개만 사용)

## 결과 기록 규칙
실험마다 "변경점 / split / pool / score / R@1 / Recall@10·20·50 / 추론 시간"을 실험 기록표에 한 줄씩.
