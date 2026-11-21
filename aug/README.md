# B 트랙 학습 도구 (aug/)

대회 규정 준수: 외부 생성 모델 없이 **규칙 기반**으로만 변형한다. 모든 옵션은 기본 꺼짐 → 하나씩 켜며 valA로 ablation.

| 파일 | 내용 | 종류 |
|---|---|---|
| `image_aug.py` | `weak_square_crop`(0.8~1.0배), `hflip` | 증강(새 정답 쌍) |
| `text_aug.py` | `swap_left_right` (반전 이미지와 짝 → 새 정답 쌍) / `hard_negatives` (색·색결합·관계·무늬·재질 변형 → 오답 캡션) | 증강 + 오답 생성 |
| `dataset.py` | `make_pair` (증강 적용한 정답 쌍 + 오답 캡션), `GroupBatchSampler` (형제를 같은 배치에) | 샘플링 |
| `loss.py` | `siglip_loss` + 텍스트 hard negative 항 | 손실 |

검증 결과 (train 2,400, 배치 32)
- 그룹 배치: 형제와 같은 배치에 들어가는 샘플 **80%** (무작위 배치 4%)
- 오답 캡션 생성 가능: 색 88% / 관계 73% / 색결합 71% / 무늬 27% / 재질 16%, 하나도 못 만드는 캡션 0.3%
- 좌우 단어 포함 43% → 반전+교환 대상
- 손실: 정답과 거의 같은 가짜 캡션일수록 손실 증가, 무관한 캡션은 영향 없음 확인

알려진 한계: 'orange'가 과일 이름일 때도 색으로 바꿀 수 있음(드묾). 'next to'는 대칭이라 뒤집지 않음.

## ablation 순서 (권장)
1. 기본 FT (무작위 배치) ← 기준
2. + GroupBatchSampler
3. + flip_swap_p=0.5
4. + n_text_neg=2 (loss의 neg_txt)
5. + crop_p=0.5
