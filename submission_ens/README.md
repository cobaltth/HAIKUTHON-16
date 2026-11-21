# submission_ens — A·B·C 3단계 앙상블

| 단계 | 하는 일 | 모델 |
|---|---|---|
| 1 | 16,000장 → 질의별 top-k1 | SigLIP Base (A 학습본, fp16) |
| 2 | top-k1 재정렬, 1단계와 z-점수 합산 (blip_weight) | BLIP-2 ITM (A 학습본, int8 저장) |
| 3 | 1·2단계 차이가 작은 질의부터 상위 k3 재판정, z-점수 합산 (qwen_weight) | Qwen3-VL-Reranker-2B + LoRA (C, int8) |

- 정규화는 모두 질의별로만 함(질의끼리 정보 공유 없음). CSV 병합 없음.
- 시간 가드: `time_limit`(predict 시작 기준) 넘으면 멈춤. 2단계는 1위권 후보 이미지부터 처리하고, 질의마다 앞에서부터 채점된 칸까지만 재정렬에 씀(나머지는 1단계 순서).
- 설정: zip의 `models/ens_config.json`이 `model.py`의 `DEFAULT_CFG`를 덮어씀. 첫 제출 설정은 `ens_config.k15_t840.json`.

## 만들기
```
python tools/build_ens.py --out sub_ens --siglip <A models/> --blip <A reranker/> --qwen <C models/reranker/> \
    --blip-q8 --siglip-fp16 --config '{"k1":15,"k3":3,"blip_weight":0.9,"qwen_weight":0.5,"time_limit":840,"stage2_deadline":600}' --zip
```

## 검증 (304 질의 × 4,000장, A 무작위 분할 valid ∩ C 그룹 valA∪valB)
| 버전 | 점수 | R@1 | R@5 |
|---|---|---|---|
| A K=20 | 0.698 | .579 | .987 |
| 앙상블 K=20 fp16 | 0.793 | .717 | — |
| 앙상블 K=15/840 int8 (제출) | 0.802 | .730 | .987 |

주의: A 모델은 무작위 분할로 학습해 형제 이미지를 봤기 때문에 이 로컬 점수는 부풀려져 있음. 테스트 판단은 리더보드로.

## 11/22 새벽 버전 (파일 안내)
| 파일 | 내용 | 제출 |
|---|---|---|
| `model_k32_blip_only.py` + `ens_config.k32_blip_fp16.json` | SigLIP(A) top32 → BLIP-2(A) 전부 재정렬. ITM fp16(autocast) + 문장 길이순 묶음. Qwen 없음 | 09 `sub_blip_k32_fp16` |
| `model.py` + `ens_config.k20_qwen2_fuse.json` | 위 구조 + Qwen(C v4, 192px, C의 비전 캐시 패치) 상위 k3. 최종 = BLIP 표준화 점수 + Qwen 로짓/8 합산 | `sub_blip_qwen_fuse_v4.zip` (미제출) |
| `model_zip_min.py` | `model.py`의 주석 제거·들여쓰기 1칸 압축본 (AST 동일). zip 안 model.py를 같은 크기(20,987바이트)로 제자리 교체하려고 만든 것 | 위 zip에 실제로 들어간 파일 |

- 규정(질문별 독립 예측) 반영: 이전 버전의 "2단계 1·2위 차이가 작은 질문부터 Qwen" 순서는 **제거**. 모든 질문을 입력 순서대로 같은 깊이까지 처리. (06 `ens_A_blip_C_qwen_k15`에는 그 순서가 들어 있었으므로 최종 선택에서 제외)
- 로컬 확인(clean 304 × 4,000, Kaggle T4): K32 BLIP만 0.6873 (R@1 .5625), 형식 검사 통과. Qwen 결합 버전은 형식 검사 통과 + 비전 캐시 자체 검증 차이 0.0000.
- 실험 근거·기각한 아이디어는 `docs/B_track_guide.md` 11/22 항목 참고.
