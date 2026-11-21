# B 트랙 (증강·임베딩) 메인 지침 — 가민서

> 이 문서를 기준으로 작업을 진행한다. 단계가 끝나면 체크하고, 결정사항·수치는 맨 아래 "진행 로그"에 적는다.

## 0. 역할 한 줄 요약
1단계 검색(임베딩)이 정답을 top-K 후보 안에, 최대한 위쪽에 넣도록 만든다.
→ 데이터 파악 → 검증셋 → 베이스라인 측정 → 백본 스윕 → hard negative FT → 전체 데이터 재학습·전달

## 고정 조건 (대회 규정·제약)
- Task: 영어 query → 이미지 top-5 순위 제출, 정답 1장. 점수 = (R@1 + NDCG@5)/2 → 1위 1.0 / 2위 0.315 / 3위 0.25 / 4위 0.215 / 5위 0.193
- Test: 텍스트 8,000 / 이미지 16,000 (비공개). Train: 4,000쌍, 768×768 PNG, `prompts.jsonl` = {id, prompts}
- 추론: L4 24GB **1장**, 15분(로딩 제외), zip ≤ 5GB, 제출 25회, 마감 11/22 08:00 (대기열 → 최종 제출 06~07시 전)
- 금지: 외부 API / 다른 query 예측으로 현재 query 보정(1:1 매칭 X) / closed-weight 생성 AI로 데이터 증강(텍스트 포함 → hard negative 문장은 **규칙 기반 코드로만**)
- 학습 GPU: Colab·Kaggle **무료** GPU만 (T4, bf16 없음 → fp16). Kaggle 권장, 체크포인트 자주 저장
- 용량: Qwen3-VL-Reranker-2B 단독 4.26GB → 1단계 백본은 작게/양자화 고려, zip 크기 팀과 공유

## 1단계: EDA (데이터 도착 즉시)  [x] 16:55 완료 (eda/ 폴더)
- [x] 캡션 30개 + 이미지 몇 장 육안 대조
- [x] 이미지 해상도·포맷·개수(4,000) 확인
- [x] 캡션 템플릿 패턴 추출 (개수/색/관계/장소/화풍/시점/조명)
- [x] 속성만 다른 캡션 쌍 존재 여부
- [x] 캡션 토큰 길이 분포 → SigLIP 64토큰 초과 비율
- [x] left/right 등 좌우 관계어 빈도 (좌우 반전 허용 여부 판단)

## 2단계: 검증셋 분할  [x] 17:05 완료 (splits/, eval/)
- [x] train 2,400 / val-A 800 (튜닝·비교용) / val-B 800 (최종 확인용), seed 고정, id 목록 파일로 저장해 팀 공유
- [x] 대회 점수식 그대로 구현한 평가 함수 (R@1, NDCG@5, score) + 순위 분포 출력
- [x] 검증 풀에 query 없는 distractor 이미지 섞어 테스트 비율(1:2) 근사

## 3단계: 베이스라인 점수 + 순위 분포 표 (17:30~19:30)  [ ]
- [ ] 베이스라인(A) 모델로 val-A 평가
- [ ] 정답 순위 분포 표: 1 / 2–5 / 6–10 / 11–20 / 20+ 비율, Recall@K (K=1,5,10,20,50)
- [ ] C(재정렬)에게 전달 → 재정렬 K 결정 근거 (게이트 판단)

## 4단계: 백본 스윕  [ ]
- [ ] 후보 임베딩 모델별 val-A 점수 / 16,000장 인코딩 시간(T4 fp16) / 용량 표
- [ ] 긴 캡션 처리 비교 (truncate vs 문장 분할 평균)
- [ ] 승자 선정 (점수·시간·용량 함께 고려)

## 5단계: hard negative 생성 + 파인튜닝 (19:30~01:00)  [ ]
- [x] (코드 준비) 텍스트 hard negative: 규칙 기반으로 색/개수/관계/물체 치환 (코드로만)
- [ ] 이미지 hard negative: 1단계가 높은 순위에 올린 오답 이미지 마이닝
- [ ] 대조학습 FT (train 2,400), 이미지 증강은 약한 크롭(0.9–1.0) + 좌우 반전(**문장의 left↔right를 같이 바꿀 때만 허용**). 색 jitter 금지
- [ ] val-A로 비교, 최종 val-B 1회 확인

## 6단계: 마무리  [ ]
- [ ] 최종 설정으로 4,000쌍 전체 재학습
- [ ] 가중치는 드라이브로 A에게 전달 (깃헙 X)
- [ ] 실험 기록표 정리 (실패 포함) → 발표 ablation 자료

## 팀 인터페이스 약속
- 1단계 출력 형식: `query_id → [(image_id, score), ...]` top-K
- 실험 기록: 변경점 / val 점수 / 추론 시간 / 용량
- 중간 합류: 19:00, 02~03시

## 진행 로그
- 16:40 지침 작성. 데이터 다운로드 중 (images 3,935장 도착, prompts.jsonl·starterkit 미도착)
- 16:45 다운로드 완료 확인 (C:\Users\bobbi\haikuthon-train: images 4,000 / prompts.jsonl / starterkit.zip). 스타터킷 분석:
  - 인터페이스: `load_model(model_dir, device)` → model, `predict(model, queries, candidates, top_k=5)` → {prompt_id: [image_id×5]}. queries={prompt_id, prompt}, candidates={image_id, image_path} → **PNG 디코딩도 추론 시간에 포함**
  - 베이스라인: google/siglip-base-patch16-224(203M), 출력 헤드만 1에포크 학습. 분할 seed 42, 3,200/800. 검증(풀 800장) R@1 0.6925 / NDCG@5 0.8752 / 평균 0.7839 (테스트 풀은 16,000장이라 훨씬 낮을 것)
  - 텍스트 max_position_embeddings=64, truncation=True → 긴 캡션 잘림 확정 / 이미지 224px로 축소
  - 결정 후보: val-A를 스타터킷 검증 800(seed 42)과 일치시켜 베이스라인 수치와 비교 가능하게
- 16:48 작업 폴더 이동: C:\Users\bobbi\Desktop\haikuthon\ 아래에 HAIKUTHON-16-(레포) / haikuthon-train(데이터) / starterkit(압축 해제본). 무결성 확인 완료: prompts 4,000행(id 중복 없음) ↔ 이미지 4,000장 1:1, 전부 768×768 PNG, 깨진 파일 0, 스타터킷 압축 해제 누락 0
- 16:55 EDA 결과 (eda/ 스크립트, eda/groups.csv)
  - 캡션 = 장면 그래프(물체·개수·색·무늬·재질·상태·관계 + 장소·화풍·시점·조명)를 문장 순서만 바꿔 서술. 어휘 587개
  - 토큰: 중앙값 42, 99% 64, 최대 77 → 64토큰 초과 0.9%뿐, 잘리는 건 주로 끝의 화풍/조명 → 긴 캡션 처리는 우선순위 낮춤 (예상과 다름)
  - 화풍 12종(사진·3D 각 ~20%), 시점 4종, 조명 4종 거의 균등, 장소는 studio 49%
  - 관계: left 21%·right 22%(좌우 합 ~43%) / next to / above / below / on / holds / inside / wears / rides. 개수 two 22%, three 1%
  - ★ 최소 차이 그룹 구조: 물체 구성이 같은 캡션끼리 묶으면 1,564그룹, 80%(3,222개)가 2~17개짜리 그룹 소속. 그룹 안에서 색 하나, 재질 하나, 관계 방향, 색-물체 결합 교환만 다름 (예 0849↔1958, 0002↔0652, 2801↔3142) → train 안에 자연 hard negative 존재. 테스트(텍스트 8,000/이미지 16,000)도 같은 구조일 가능성 큼
  - 결정 후보: (1) 검증 분할은 그룹 단위로 (형제가 같은 쪽에 있어야 현실적인 오답 포함, 누수 방지) (2) FT 배치를 그룹 단위로 구성 → 규정 위반 없는 hard negative (3) 좌우 반전 + 캡션 left↔right 교환 = 정답 쌍 증강 / 반전 이미지 = 원 캡션의 hard negative
- 17:05 그룹 단위 분할 완료: splits/split_group_seed42.json (2,400/800/800, 형제 분리 0, 세트별 그룹 크기 분포 유사). 비교용 splits/split_random_starterkit.json(스타터킷과 동일)
  - 스타터킷 무작위 분할은 검증 캡션의 79%가 train에 형제를 둠 → 검증이 테스트와 다른 조건일 가능성
  - eval/harness.py: 제출 폴더 model.py 그대로 채점, --pool pair(valA+valB 1,600장, 1:2 비율) 기본. 가짜 모델로 점수식 검증 완료
  - 다음: 첫 제출 LB 점수 vs (group/pair) vs (random/own) 비교로 "테스트에 형제 이미지 있음" 가설 확인
- 17:11 정정: A의 첫 제출은 sample이 아니라 팀 자체 모델. 대회 공지 sample_submission LB = 0.2319 (스타터킷 자체 검증 0.784와 큰 차이 → 후보 16,000장 + 형제 이미지 영향 추정)
  - 주의: sample은 스타터킷 무작위 3,200쌍으로 학습 → 우리 valA의 약 80%를 학습에서 봄. sample은 random split val로만 평가할 것. A 모델도 학습 분할 확인 필요 (평가용은 group train 2,400으로)
- 17:35 학습 도구 aug/ 작성·검증 (GPU 불필요 부분 완료, 실제 학습은 3·4단계 후)
  - GroupBatchSampler: 배치 32 기준 형제와 같은 배치 샘플 80% (무작위 4%) ← 발표용 수치
  - 텍스트 오답 생성 가능: 색 88% / 관계 73% / 색결합 71% / 무늬 27% / 재질 16%
  - 반전+left↔right 교환(정답 쌍), 약한 크롭, SigLIP 손실+텍스트 오답 항(동작 검증 완료)
  - ablation 순서: 기본 FT → +그룹배치 → +반전교환 → +텍스트오답 → +크롭
- 17:40 리뷰 반영 (eval/): 이미지 경로 문자열 절대경로로 전달 / 채점 전 top_k=5 서버 형식 검사(중복·개수·후보 외 id) / 스타터 가중치+그룹 분할 조합 경고 / eval/compare.py(paired bootstrap 95% CI 채택 판정) 추가. 가짜 모델 3종으로 테스트 통과
- 17:45 2번(육안 점검) 완료, viz/ 저장: 반전+교환 4/4 캡션 일치(이미지 내 글자 없음 → 안전). 크롭 0.8은 가장자리 물체 반쯤 잘림 2건(0572, 1060) → 기본 0.9로 상향, ablation 최하위. 다음: 1번 학습 스크립트(어댑터 구조), A에게 백본·분할·학습층·해상도 문의
- 18:00 학습 스크립트 완료 (train/): train_b.py(어댑터 구조, 그룹 분할, 증강 옵션 전부 기본 꺼짐, 스타터킷 predict로 매 에포크 valA 채점, best 저장, runs/<name>.json → compare.py), adapters.py(SiglipAdapter, 새 백본은 클래스만 추가), run_ablation.sh(r0~r5 누적 + 자동 비교), KAGGLE.md(셀 순서). submission_template/ = 스타터킷 model.py
  - 스모크 테스트(CPU, 작은 랜덤 SigLIP): 학습→평가→저장→하네스 재채점 점수 일치(0.2003), 텍스트 오답 실제 투입 확인
  - 출발 가중치는 google/siglip-base-patch16-224 (스타터킷 예제 가중치는 valA 오염)
  - 다음: Kaggle에서 r0_zeroshot 먼저 → run_ablation.sh
- 17:57 규칙 변경 반영: "좌우 반전 금지" → "좌우 반전은 문장의 left↔right를 같이 바꿀 때만 허용". 코드상 hflip은 항상 swap_left_right와 함께만 호출됨(train_b.py, aug/dataset.py) 확인
- 18:20 ★ r0_zeroshot (Kaggle T4, google/siglip-base-patch16-224 학습 없음, valA 800 / 후보 1,600): score 0.5166, R@1 0.365, NDCG@5 0.668, Recall@5 0.934, Recall@10 1.000
  - 정답 순위 분포: 1위 36.5% / 2-5위 56.9% / 6-10위 6.6% / 10위 밖 0%  → 정답은 항상 top-10 안, 문제는 1위 다툼(형제 구분) → 재정렬 K≈10이면 valA 기준 100% 커버 (테스트는 후보 10배라 더 필요할 수 있음)
  - ⚠ 검증 추론 88.7s 대부분이 이미지 1,600장 디코딩·인코딩 → 테스트 16,000장이면 이미지 쪽만 단순 10배 ≈ 15분 위험. model.py 이미지 로딩 병렬화 필요 여부 측정 필요 (A·C 공유)
- 18:30 r0 오답 분석: 1위 틀린 508개 중 366개(72%)는 1위를 형제 이미지가 차지. 개수만 다른 형제(복수형 단어 차이, 예 0015↔2162)는 그룹에서 빠져 있어 72%는 하한. 예시 viz/r0_errors.png (드라이버 색, 새 색, 색-물체 결합 교환, 위아래 관계, 개수)
- 18:45 ★ 후보 수 스케일링 (eval/pool_scaling.py, zero-shot SigLIP Base, 캡션 4,000개 전부 질문, 그룹 단위 샘플링)
  - 실측 score / R@1 / R@5 / R@10:  800장 .568/.432/.938/1.000 · 1600장 .523/.378/.922/.998 · 2000장 .497/.349/.911/.997 · 4000장 .425/.274/.858/.993
  - 외삽 검증(4000→2000 예측): R@5 .913 vs 실측 .907, R@10 .998 vs .997 → R@5·R@10은 잘 맞음. R@1은 방식상 4000장 값에 고정되는 한계(예측 불가)
  - 16,000장 추정 (낙관=형제 수 유지 / 비관=형제도 4배): R@10 .80/.65, R@20 .95/.86, R@30 .99/.99, R@50 .998/.997
  - 필요 K: 90% 커버 15~22, 95% 20~26, 98% 26~30, 99% 30~34
  - R@1 추세(로그 스케일 외삽): 후보 2배마다 약 -0.08 → 16,000장 R@1 ≈ 0.12, score ≈ 0.30 추정 (sample LB 0.2319와 비교용)
  - 시간(Kaggle T4, 장당): 단일 스레드 디코딩 20.9ms(16,000장 5.6분) / 병렬 디코딩 8.0ms(2.1분) / 전처리 5.5ms(1.5분) / GPU 2.4ms(0.6분) → 스타터킷 방식 임베딩 단계 ≈ 7.7분, 병렬화 시 ≈ 4분. 앞서 '15분 위험'은 과대추정이었음(정정)
- 19:00 (추가 역할) SigLIP + VLM 재정렬 베이스라인 작성
  - submission_vlm/model.py: 1단계 SigLIP(PNG 병렬 디코딩, 묶음 단위 프리페치로 메모리 제한, VLM용 448px JPEG 캐시) → 2단계 Qwen3-VL-Reranker-2B로 상위 K 재정렬(공식 코드와 같은 프롬프트·yes-no 점수, 배치 처리). 설정은 models/config_b.json (k, vlm_size, vlm_batch, time_budget_sec=780, margin_threshold, fuse_alpha)
  - 시간 가드: 경과 시간만 보고 재정렬 중단 → 남은 질문은 SigLIP 순위 (다른 질문 결과는 참고 안 함 = 규정 준수)
  - tools/prepare_submission.py: SigLIP fp16 저장 + VLM 다운로드 + 설정 + 용량 점검 + 형식 자체 테스트 + zip(무압축)
  - CPU 스모크 테스트(SigLIP 단독 경로): 형식 통과, 학습 스크립트 평가와 점수 일치. VLM 경로는 Kaggle에서 시험 예정(ablation 끝난 뒤)
  - ablation 중간: r1_base 0.5343 (r0 0.5166), r2_group 진행 중
- 19:05 제출 사이트(2026-haikuthon.pages.dev, 팀 group16) 안내 사항 확인. 우리 팀 제출 0/25
  - ★ 평가 환경: L4 24GB 1장, **CPU 4 vCPU, RAM 16GiB**, Python 3.12.16, torch 2.15.0+cu132 (변경 불가), transformers 5.22.2 기본 설치
  - requirements.txt로 PyPI wheel 추가 설치 가능 (PackageName==Version만, 설치 3분·1GB 이내, 설치 시간은 추론 시간에 미포함). 기본 제공: transformers, huggingface-hub, sentence-transformers, open-clip-torch, timm, scipy, einops 등
  - 추론 시간 = predict() 실행 시간. **15분 초과 실패는 1회 차감**, 그 외 실패는 차감 없음. 같은 팀 제출이 대기·평가 중이면 새 제출 불가
  - device 인자 예: "cuda:0". predict는 프롬프트 8,000 + 이미지 16,000으로 한 번 호출, top_k=5 고정, 서로 다른 후보 id 5개
  - 금지: 외부 데이터(학습·검증·추론), 외부 API, 테스트 데이터로 추가 학습, **테스트 예측 분포로 최종 예측 조정**. 위반 시 무효 + 1회 차감. 사이트 생성 CSV 수정 후 Kaggle 제출 금지
  - 허용: 사전학습 모델 자유 사용. 학습 데이터 증강·생성형 AI 변형 데이터 허용이라고 적혀 있으나, 카톡 공지(closed-weight 생성 AI 증강 금지)가 더 구체적·최신 → 공지 기준 유지
  - Public 3,200 / Private 4,800, 최종 리더보드 = 전체 8,000 평균
  - 베이스라인 반영할 것: decode_workers 4 (vCPU 4), RAM 16GB 안(JPEG 캐시 ~0.7GB OK), 시간 가드 유지(초과 시 차감)
- 19:25 ★ ablation 결과 (SigLIP Base, 헤드만 학습, 3에포크, valA 800 / 후보 1,600, 앞 단계 대비 paired bootstrap)
  - r0 zeroshot 0.5166 → r1 base 0.5343: +0.0177 [95% CI +0.0021, +0.0333] **채택**
  - r1 → r2 +형제배치 0.5249: −0.0094 [−0.0216, +0.0024] 차이 불명확 (오히려 하락 경향)
  - r2 → r3 +반전교환 0.5233: −0.0017 [−0.0087, +0.0051] 불명확
  - r3 → r4 +가짜캡션 0.5248: +0.0015 [−0.0068, +0.0097] 불명확
  - r4 → r5 +크롭 0.5251: +0.0003 [−0.0059, +0.0068] 불명확
  - 해석: 헤드만 학습하는 설정에서는 기본 FT만 효과. 증강·hard negative 효과 없음(누적 r5 0.5251 < r1 0.5343). 헤드(마지막 투영)만으로는 결합·관계를 배울 용량이 부족하다는 가설 → 다음: --trainable last2 / 더 긴 학습으로 r1 vs +group 재비교
- 19:45 ★ VLM 베이스라인 첫 실측 (Kaggle T4, submission_vlm/model.py md5 7d2d6e0a…, sub_vlm_k5 용량 4.68GB)
  - dtype 수정: T4(sm75)에서 bf16이 에뮬레이션돼 1쌍/초 미만 → GPU 세대 기준(sm80+ bf16, 그 외 fp16)으로 변경. fp16 NaN 없음
  - valA(800/후보 1,600), K=5, VLM 448px, 배치 20: **score 0.7303 (SigLIP 단독 r0 0.5166 대비 +0.214)**, R@1 0.644 (0.365), NDCG@5 0.817, Recall@5 0.934(=SigLIP 상한), 정답 순위 1위 64.4% / 2-5위 29.0% / 5위 밖 6.6%
  - 속도: 1단계 1,600장 27.3s (17ms/장, JPEG 캐시 포함), 2단계 4,000쌍 520s = **7.7쌍/s (T4 fp16)**
  - 테스트 환산(T4): 1단계 16,000장 ≈ 4.5분, 2단계 40,000쌍 ≈ 87분 → 15분 불가. L4 2~2.5배 가정해도 시간 가드 내 재정렬 가능 ≈ 8~10k쌍 → 대안 측정 중 (K=3, 224/336px)
- 20:05 ★ VLM 설정별 valA 비교 (T4 fp16, 후보 1,600, runs/vlm_*.json)
  | 설정 | score | R@1 | 2단계 속도 |
  |---|---|---|---|
  | SigLIP 단독 (r0) | 0.5166 | 0.365 | - |
  | K=5, 448px, 배치20 | **0.7303** | 0.644 | 7.7쌍/s |
  | K=3, 336px, 배치36 | 0.6732 | 0.579 | 11.7쌍/s |
  | K=3, 224px, 배치48 | 0.6569 | 0.556 | 17.1쌍/s |
  - 테스트 환산(8,000질문): 1단계 16,000장 ≈ 4.5분(4 CPU, 장당 ~16ms). 2단계 쌍 수 K=3 → 24,000 / K=5 → 40,000
    - T4 기준 2단계: K3·224 ≈ 23분, K3·336 ≈ 34분, K5·448 ≈ 87분 → 전부 시간 가드에 걸림
    - L4를 T4의 2~2.5배로 보수 가정: 2단계 예산 ≈ 510초(780−270) → K3·224면 질문의 약 70~90%, K3·336이면 약 50~60%만 재정렬, 나머지는 SigLIP 순위
  - 권장(첫 VLM 제출): K=3, 224px, 배치48, time_budget 780 유지 (초과 시 1회 차감이라 여유 유지). 해상도보다 '재정렬되는 질문 비율'이 점수에 더 중요
  - 추가 가속 후보: 224px에선 이미지 토큰(~49)보다 텍스트(시스템+지시문+캡션 ~170토큰)가 커서 지시문·시스템 프롬프트 축약만으로 ~1.3~1.5배 기대 / margin_threshold로 확실한 질문 건너뛰기 / K3·448 미측정
  - 현재 /kaggle/working/sub_vlm_k5/models/config_b.json = k3·336 상태 (마지막 실험값)
- 20:00 ★ 최종 제출용 VLM zip 생성 (Kaggle): /kaggle/working/sub_vlm_k3_224.zip, 56파일, 약 4.68GB(5GB 이내), model.py 최상위
  - 설정: rerank=true, k=3, vlm_size=224, vlm_batch=48, time_budget_sec=780, SigLIP google/siglip-base-patch16-224(학습 없음, fp16) + Qwen/Qwen3-VL-Reranker-2B (L4에서는 bf16, T4는 fp16 자동)
  - 자체 형식 검사 통과 (T4 fp16, 18.2쌍/s). valA 전체 재정렬 시 0.6569 / 테스트에선 시간 가드로 일부만 재정렬 예상
  - 프롬프트 단축 실험은 보류 (민서 결정)
- 21:00 ★ 사이트 업로드 거절("ZIP 업로드 크기 제한을 초과했습니다", 4.68GB zip) → VLM int8 양자화 제출본 생성
  - tools/quantize_vlm.py (md5 2ddc2a76…): VLM 2차원·100만 원소 이상 가중치 301개를 int8 + 행별 배율로 저장(pos_embed 등 나머지 324개 원래 dtype), 묶인 가중치 1개. 상대오차 ~1~2%
  - submission_vlm/model.py (md5 53cc58f7…): models/vlm/model_q8.safetensors가 있으면 load_model에서 bf16/fp16으로 복원 (추론 시간 미포함)
  - 결과: /kaggle/working/sub_vlm_k3_224_q8.zip **2.56GB** (2,559,050,909 bytes), SigLIP 389MB + VLM 2.1GB
  - valA(K=3, 224px, T4 fp16): **0.6586** (원본 0.6569), R@1 0.5587, NDCG@5 0.7584, 17.2쌍/s → 양자화 손실 없음
  - 서버 실제 업로드 한도는 미확인 → 운영진 문의 권장
- 23:47 ★ 첫 LB 결과 vlm_k3_224_v1 (int8 zip 2.56GB): public avg **0.2492** (R@1 0.1872, NDCG@5 0.3112, R@5 0.4319), predict 778.8s(시간 가드 780s 도달 → 일부 질문만 재정렬). BASELINE 0.2319 대비 +0.017
  - R@5 0.432 = SigLIP 1단계 R@5 (재정렬은 top3 안 순서만 바꿈) → valA 0.934·4000장 외삽보다 훨씬 낮음. 병목은 1단계 재현율
  - CSV 분포: top-1 고유 이미지 3,782개/8,000질문(최대 한 이미지가 21질문 1위), top-5 전체 고유 8,568장/16,000 → 허브 이미지 문제. 단 테스트 예측 분포로 조정은 규정 위반 → 테스트 기반 허브 보정 금지
- 23:55 add-one ablation 시작 (Kaggle 새 세션, T4): r0 + a1_last2(기준) + a2 group / a3 flip / a4 textneg / a5 crop 각각 a1 대비 (run_addone.sh, /kaggle/working/addone.log)
- 00:00 ★ A(태환) LB: CLIP+VLM_v1 public avg **0.3275** (R@1 0.2725, NDCG@5 0.3826, R@5 0.4644), predict 386s. 우리 B 0.2492 대비 +0.078
  - R@5 0.464 > B 0.432 → 학습한 SigLIP(헤드 1에포크) 1단계 재현율이 zero-shot보다 높음
  - 5위 안 정답 중 1위 비율 A 59% vs B 43% → 학습한 BLIP-2 ITM + 점수 결합이 zero-shot Qwen보다 강함, 시간도 386s로 전부 재정렬
  - 두 CSV 비교: top-1 일치 35.3%, 평균 top-5 겹침 3.44/5, A top-1 고유 이미지 4,641 (B 3,782)
- 00:05 add-one 중간: a1_last2(마지막 2층 학습, 증강 없음) 0.5640 (r1 헤드만 0.5343, r0 0.5166). a6_shuffle(문장 순서 섞기, aug/text_aug.py shuffle_sentences + --sent-shuffle-p) 대기열 추가
- 00:30 ★ add-one ablation 결과 (SigLIP Base, last2, 3에포크, valA 800/후보 1,600, a1 대비 paired bootstrap)
  - r0 0.5166 → a1_last2 0.5640: +0.0474 [+0.0274, +0.0676] **채택** (r1 헤드만 0.5343보다 확실히 좋음)
  - a2 group-batch 0.5376: −0.0265 [−0.0427, −0.0103] **기각 (유의하게 나쁨)**
  - a3 flip-swap 0.5638: −0.0002 [−0.0066, +0.0065] 불명확 / a4 text-neg 0.5552: −0.0088 [−0.0248, +0.0067] 불명확 / a5 crop: −0.0013 [−0.0086, +0.0062] 불명확
  - 결론: SigLIP 1단계에는 증강 효과 없음 → last2 학습만 채택, group-batch 제외
- 00:30 텍스트 TTA (eval/text_tta.py, zero-shot SigLIP, 문장 순서 섞기 버전 평균): 효과 없음
  - valA/1,600: v1 0.5166 / v2 0.5175 / v4 0.5157 / v8 0.5162 (평균 서로 다른 버전 수 최대 2.4개)
  - 전체 4,000/4,000: v1 0.4252 R@5 .858 / v2 0.4223 / v4 0.4215 / v8 0.4213 → 약간 하락. 채택 안 함
- 00:30 C(minprompt) LB 0.3038 (R@1 .257, R@5 .434, 780s). CSV 비교: A↔C top-1 일치 22.8%, top-5 평균 겹침 1.87 (A↔B 3.44) → C는 A와 다른 후보를 잡음. A·C 중 하나라도 1위 정답 ≥ 0.30. CSV 합치기는 규정 위반 → model.py 안에서 앙상블해야 함
- 00:37 a6_shuffle(문장 순서 섞기, last2) 0.5660: a1 대비 +0.0020 [−0.0080, +0.0123] 불명확 → 증강 전부 효과 없음으로 정리
- 00:38 C 코드 확인(C-reranker v4): SigLIP2-so400m@256 1단계 + Qwen3-VL-Reranker-2B LoRA(int8, 프롬프트 없음, 20ms/쌍 L4), 질문별 k(gap 0.05, 최대 6), 랭크 순서 재정렬 + 780s 가드, 한 번 디코딩. 로컬 valA 0.849 / LB 0.3038
- 00:40 결정(안): A를 뼈대로 3단 깔때기 — [1] 학습 SigLIP Base(last2, 4,000쌍) top-K1 20~30 → [2] A의 BLIP-2 ITM으로 top-3 → [3] C의 Qwen LoRA로 애매한 질문 1~3위 재판정(780s 가드), 최종 = BLIP-2+Qwen z-가중합. 용량 ≈ 3.8GB(BLIP-2 int8)
  - 1단계 앙상블 측정 시작: eval/stage1_fusion.py (valA 800 × 후보 4,000, Base 0-shot / Base a1 / SigLIP2 so400m@256 및 쌍별 z-가중합), /kaggle/working/fuse.log

## 팀 가중치 위치 (마감까지 사용)
- A(태환, CLIP+VLM_v1 등): https://drive.google.com/drive/folders/1VqjSIPzbnRkDsMNe1cDoMsfF_dR7Noic
- C(연오, minprompt/v4 등): https://drive.google.com/drive/folders/1PWUw4dB3CEzVvVU4TtAGaLrcpCpS9t69
- Kaggle에서 받기: `gdown --folder <링크>` (링크 공개 설정 필요), 큰 파일은 /kaggle/working 대신 /tmp에 풀기(작업 폴더 19.5GB 한도)
- 00:50 ★ 1단계 앙상블 측정 (eval/stage1_fusion.py, valA 800 × 후보 4,000; a1은 train 이미지를 학습에서 봤으므로 약간 낙관적)
  | 1단계 | score | R@1 | R@3 | R@5 | R@8 | ms/장(T4) |
  |---|---|---|---|---|---|---|
  | SigLIP Base 0-shot | 0.431 | 0.276 | 0.675 | 0.869 | 0.991 | 3.1 |
  | **SigLIP Base a1(last2 학습)** | **0.478** | **0.321** | 0.735 | **0.911** | 1.000 | 3.1 |
  | SigLIP2 so400m@256 0-shot (C) | 0.420 | 0.266 | 0.678 | 0.847 | 0.990 | 19.3 |
  | a1×0.6 + so400m×0.4 (z 가중합) | 0.470 | 0.315 | 0.738 | 0.894 | 1.000 | – |
  | a1×0.7 + so400m×0.3 | 0.469 | 0.310 | 0.750 | 0.904 | 1.000 | – |
  - 결론: 학습한 SigLIP Base 단독이 최고. SigLIP2를 섞으면 R@3만 조금 오르고 점수·R@5는 하락, 인코딩 6배 느림 → 3단 깔때기 1단계는 학습 Base 단독 (s2_weight 0)
- 00:50 submission_ens/model.py(3단 깔때기) + tools/build_ens.py 작성, harness에 --pool all 추가. 학습 안 된 BLIP-2/Qwen으로 시간·동작 시험 시작 (/kaggle/working/ens.log)
- 00:50 A 드라이브 폴더는 로그인 요구(비공개) → '링크가 있는 모든 사용자' 공유 필요. C 폴더는 공개 접근 가능
- 01:15 ★ A K=20 제출(SIGLIP+BLIP-2_v2) LB 0.2319로 하락 (R@1 .153, R@5 .464, 721.4s)
  - 원인: A model.py는 time_budget 720s(load_model 시작 기준)에 걸리면 재정렬 중단 + '20칸 중 하나라도 채점 안 된 질문은 통째로 1단계 순위' + 이미지를 번호순으로 처리 → K=20이면 거의 모든 질문이 미완료 → 사실상 SigLIP 단독(R@1 .153). R@5가 v1과 같은 .464인 것도 1단계 그대로라는 증거
  - 처방: (1) 이미지를 '가장 높은 순위' 순으로 처리 (2) 앞에서부터 채점된 칸까지만 융합(부분 재정렬) (3) K 10 안팎 / BLIP 속도 측정 후 결정
  - submission_ens/model.py에 (1)(2) 반영 완료
- 01:15 팀 clean 검증(A 랜덤분할 valid ∩ C 그룹분할 valA∪valB = 304 질문 × 후보 4,000): A K=20 0.6984 (R@1 .579, R@5 .987) — 로컬은 시간 가드 안 걸림
- 01:30 ★★ 팀 앙상블 검증 (clean 304 질문 × 후보 4,000, Kaggle T4)
  - A K=20 (A model.py): 0.6984 (R@1 .579, R@5 .987)
  - ENS = A K=20 → BLIP-2(w .9) → C Qwen LoRA 상위3 (w .5): **0.7930** (R@1 .717) → +0.0946 [+0.0655, +0.1241] **채택**, 질문별 47승 5패
  - 단, T4에서 Qwen이 912쌍 중 240쌍(120질문)만 처리하고 시간 가드 → Qwen 속도 비정상(≈2.4s/쌍) 점검 필요. 그래도 +0.095
  - 시간(T4): load 26s / decode 57s / stage1 10s / stage2(BLIP) 148s / stage3 574s(가드)
- 01:30 1단계 비교 (clean 304 × 4000): A SigLIP 0.4222 (R@5 .888) vs a1 last2 0.4535 (R@5 .924) → a1 방식이 우세, 섞으면 오히려 하락 → 1단계 교체 후보
- 01:30 제출용 sub_ens_k15_t840 빌드: 3.87GB (siglip fp16 0.41 / blip int8 1.29 / qwen int8 2.17), 검증 진행 중

## 11/22 01:55 현황 점검
- 리더보드: 16팀 16위 0.3275 (A CLIP+VLM_v1). 1위 7팀 0.4133, realbaseline 0.3915. 제출 5/25.
- 우리 제출: v1(B) .2492 / A v1 .3275 (386s) / C minprompt .3038 / A K=20 .2319 (fallback) / C v6a .3212 (R@5 .476, 799s).
- CSV 비교: A K=20과 A v1은 top-5 집합 99.9% 동일, 1위만 다름 → K=20은 사실상 SigLIP 순서 그대로. C와 A의 top-5는 평균 1.9장만 겹침 → 1단계 후보 합치기 여지.
- 테스트 R@1/R@5 ≈ .59 (BLIP·Qwen 동일) → 병목은 1단계 재현율(R@5 ≈ .46).
- 로컬 clean 0.80 vs 테스트 0.3 대: A 모델은 무작위 분할로 학습해 형제 이미지를 봤기 때문에 로컬 점수가 부풀려짐.
- T4 Qwen 느림 원인: bf16 (T4는 bf16 미지원 → 2.37s/pair). L4는 bf16 지원이라 서버는 정상(~20ms).
- 태환 님: starter_kit_kimth_v1 (4000장 검증, 5위 단위 묶음 재정렬, 840초) 재학습 중.
