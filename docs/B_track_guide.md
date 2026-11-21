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
