# submission_ens — A·B·C 3단계 앙상블

| 단계 | 하는 일 | 모델 |
|---|---|---|
| 1 | 16,000장 → 질문별 top-k1 | SigLIP Base (A 학습본, fp16) |
| 2 | top-k1 전부 재정렬, 1단계와 질문별 z-점수 합산 (`blip_weight`) | BLIP-2 ITM (A 학습본, int8 저장). 선택: 쌍둥이 세트로 추가 학습한 LoRA (`blip_lora`) |
| 3 | 모든 질문의 2단계 상위 k3를 입력 순서대로 재판정. 최종 = 2단계 점수 + Qwen 로짓/8 | Qwen3-VL-Reranker-2B + LoRA (C v4, int8, 192px, C의 비전 캐시) |

- 정규화·순위는 모두 질문 하나의 점수만 쓴다(질문끼리 정보 공유 없음). 처리 순서도 입력 순서 그대로. CSV 병합 없음.
- 시간 가드: `stage2_deadline`, `time_limit`(predict 시작 기준)을 넘으면 그 단계를 멈추고 앞 단계 순위를 쓴다. 정상적으로는 전부 끝나도록 k1·k3를 잡는다(안전장치).
- 설정: zip의 `models/ens_config.json`이 `model.py`의 `DEFAULT_CFG`를 덮어쓴다.

## 제출 기록 (이 폴더 코드로 낸 것)
| 번호 | 사이트 제출명 | zip 안 model.py | zip 안 설정 | LB | R@1 / R@5 | 시간 |
|---|---|---|---|---|---|---|
| 06 | `ens_A_blip_C_qwen_k15` | 커밋 65ff803의 `model.py` (md5 33c6b53b…) | `ens_config.k15_t840.json` | 0.3231 | .273 / .477 | 841s |
| 09 | `sub_blip_k32_fp16` | `model_k32_blip_only.py` (md5 407e2334…) | `ens_config.k32_blip_fp16.json` | 0.4372 | .321 / .753 | 655s |
| 11 | `blip_k24_qwen3_fuse` | `model_zip_min.py` (md5 4a9e0386…, 읽기용 원본 `model_v4_fuse.py`) | `ens_config.k20_qwen2_fuse.json` | 0.4804 | .396 / .719 | 738s |
| 13 | `lora_k28_qwen2_fuse` | `model.py` (v5, md5 c2d80207…) + LoRA b | `ens_config.lora_b_k28_qwen2.json` | 0.5305 | .438 / .784 | 811s |
| 15 | `lora_c_scene_k20_qwen3` | `model.py` (v5) + LoRA c + Qwen scene | `ens_config.lora_c_scene_k20_qwen3.json` | (결과 대기) | | 예상 818s |
| — | `lora_c_scene_k28_qwen2` (07:20 기준 미제출) | `model.py` (v5) + LoRA c + Qwen scene | `ens_config.lora_c_scene_k28_qwen2.json` | — | | 예상 812s |

- **11번 이름 주의.** 사이트에 적은 이름은 `blip_k24_qwen3_fuse`지만, 올린 zip(`sub_blip_qwen_fuse_v4.zip`) 안의 실제 설정은 **k1=20, k3=2, stage2_deadline 620, time_limit 838**이다. k1=24·k3=3 설정으로 낸 제출은 없다(이름만 이전 계획대로 적은 것). 그래서 `ens_config.k24_...` 파일은 없고, `ens_config.k20_qwen2_fuse.json`이 11번의 설정이다.
  - 확인 방법: zip 안 `models/ens_config.json`은 같은 JSON 뒤에 공백을 붙여 114바이트로 맞춘 것(CRC32 `e704fd47`), `model.py`는 `model_zip_min.py`와 같은 바이트(20,987바이트, CRC32 `7f5c404c`).
- 06번에는 3단계에서 "2단계 1·2위 차이가 작은 질문부터" Qwen을 돌리는 순서가 들어 있었다. 운영진 답변(11/22 04:15, "한 질문의 결과가 다른 테스트 항목의 예측 때문에 달라지면 안 됨") 기준으로 걸리므로 **최종 선택에서 제외**. 09·11번에는 그 순서가 없다.
- 06·09·11번의 가중치 파일은 같다(같은 zip에서 model.py와 설정만 제자리 교체). 09번은 zip에 Qwen이 들어 있지만 `qwen_weight: 0`이라 불러오지 않는다.
- 13번은 여기에 BLIP LoRA b(`models/blip/lora_ft.safetensors`)를 더한 것, 15번은 LoRA c를 넣고 Qwen 가중치를 scene으로 바꾼 것. 목록은 `WEIGHTS.md`.
- **만들어 두고 내지 않은 zip.** `sub_final2_c_scene_k28_qwen2.zip`(3,886,200,216바이트)은 15번 zip에 설정만 K28·상위 2장(`ens_config.lora_c_scene_k28_qwen2.json`)으로 덧붙인 것이다. `model.py`와 가중치는 15번과 같다. 13번과 같은 배분(K28·상위 2장)이라 실행 시간은 약 812초로 예상한다. 11/22 07:20 기준 제출하지 않았다.
- 12번 `sub_ens5_k22k3_scene_r192_t840`(0.5181)과 14번 `SIGLIP+BLIP-2_v4`는 다른 트랙(C, A)의 제출이라 이 표에 없다. 12번은 이 폴더 코드에 Qwen scene을 넣은 C의 버전.

## 파일
| 파일 | 내용 |
|---|---|
| `model.py` | 현재 코드(v5) = v4 + BLIP LoRA 병합. `models/blip/<blip_lora>` 파일이 있으면 로딩 때 `W += scale·B@A`로 합친다(추론 시간 변화 없음). 파일이 없거나 `"blip_lora": ""`이면 v4와 같은 동작 |
| `model_v4_fuse.py` | 11번 제출의 읽기용 원본(v4): K 전부 BLIP 재정렬(fp16) + Qwen 상위 k3 로짓 합산 |
| `model_zip_min.py` | v4의 주석 제거·들여쓰기 1칸 압축본(AST 동일). zip 안 model.py를 같은 크기로 제자리 교체하려고 만든 것. 11번 zip에 실제로 들어간 파일 |
| `model_k32_blip_only.py` | 09번 zip에 실제로 들어간 파일 (BLIP만, Qwen 없음) |
| `ens_config.*.json` | 제출별 설정 (위 표) |
| `../tools/zip_append.py`, `../tools/zip_replace_inplace.py`, `../tools/check_scene.py` | zip에 작은 파일 덧붙이기 / 같은 크기의 큰 파일 제자리 교체 / Qwen 가중치 구조 비교 |
| `WEIGHTS.md` | zip 안 가중치 파일의 크기·CRC32·출처. 가중치 자체는 용량 때문에 git에 없음 |

## 만들기
```
# 처음부터 조립 (Kaggle)
python tools/build_ens.py --out sub_ens --siglip <A models/> --blip <A reranker/> --qwen <C models/reranker/> \
    --blip-q8 --siglip-fp16 --config '{"k1":15,"k3":3,"blip_weight":0.9,"qwen_weight":0.5,"time_limit":840,"stage2_deadline":600}' --zip

# 이미 있는 zip에 파일을 덧붙이거나 바꾸기 (3.9GB 전체를 다시 쓰지 않음)
python tools/zip_append.py sub.zip model.py=submission_ens/model.py models/ens_config.json=<설정.json> \
    models/blip/lora_ft.safetensors=<LoRA 파일> --drop __pycache__/model.cpython-313.pyc

# Qwen 가중치처럼 큰 파일은 같은 크기일 때만 제자리 교체 (덧붙이면 옛 파일이 zip에 남아 5GB를 넘음)
python tools/check_scene.py sub.zip <새 int8.safetensors>          # 텐서 이름·형식·크기가 같은지 먼저 확인
python tools/zip_replace_inplace.py sub.zip models/qwen/int8.safetensors <새 int8.safetensors>
```

## 검증
(1) 304 질문 × 4,000장 (A 무작위 분할 valid ∩ C 그룹 valA∪valB)
| 버전 | 점수 | R@1 | R@5 |
|---|---|---|---|
| A K=20 | 0.698 | .579 | .987 |
| 앙상블 K=20 fp16 | 0.793 | .717 | — |
| 앙상블 K=15/840 int8 (06) | 0.802 | .730 | .987 |
| K32 BLIP만, ITM fp16 (09) | 0.687 | .563 | — |

주의: A 모델은 무작위 분할로 학습해 형제 이미지를 봤기 때문에 이 로컬 점수는 부풀려져 있다. 테스트 판단은 리더보드로.

(2) BLIP LoRA 검증: 학습에 안 쓴 8장 세트 40개(320 질문) × 후보 4,000장, `model.py`(v5)
| 버전 | 점수 | R@1 | R@5 |
|---|---|---|---|
| LoRA b, K32 BLIP만 | 0.844 | .781 | .994 |
| LoRA b + Qwen v4 상위 2장 (K20) | 0.912 | .881 | .994 |
| LoRA c + Qwen v4 상위 2장 (K20) | 0.922 | .894 | .994 |
| LoRA c + Qwen v4 상위 3장 (K20) | 0.929 | .903 | .994 |

세트 안 8지선다(같은 320 질문): 학습 전 0.692 (R@1 .566) → LoRA b 0.846 (R@1 .784) → LoRA c 0.892 (R@1 .850). 학습 코드 `train/blip_ft.py`(b), `train/blip_ft2.py`(c), 기록 `docs/B_track_guide.md` 11/22 04:30·05:20 항목.

주의: 이 로컬 수치는 8장 세트 기준이라 테스트(32장 세트)보다 훨씬 높게 나온다. 13번은 로컬 0.912, 리더보드 0.5305.

- 실험 근거·기각한 아이디어는 `docs/B_track_guide.md` 11/22 항목 참고.
