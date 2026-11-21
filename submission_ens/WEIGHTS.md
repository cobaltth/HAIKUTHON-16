# 가중치 목록 (가중치 자체는 git에 없음)

제출 zip은 3.87GB라 git에 올리지 않았다(GitHub 파일 한도 100MB). 대신 zip 안 파일의 크기와 CRC32(zip 목차에 적힌 값)를 남긴다. 같은 파일인지 확인할 때 쓴다.

- 대상 zip: `sub_blip_qwen_fuse_v4.zip` (3,866,678,028바이트, 무압축) = 제출 11 `blip_k24_qwen3_fuse`
- 제출 06·09·11은 같은 zip에서 `model.py`와 `models/ens_config.json`만 제자리 교체한 것이라 아래 가중치 파일은 셋 다 같다.

| zip 안 경로 | 바이트 | CRC32 | 출처 |
|---|---|---|---|
| `models/siglip/model.safetensors` | 406,360,284 | `7f62f2da` | A v1 제출(02 `CLIP+VLM_v1`)의 SigLIP Base 학습본, fp16 |
| `models/blip/model_q8.safetensors` | 1,284,342,816 | `a9388d01` | A v1 제출의 BLIP-2 ITM을 `tools/build_ens.py --blip-q8`로 int8(행 단위 배율) 저장 |
| `models/blip/q8_meta.json` | 43,402 | `351dccb2` | 위 int8을 되돌리는 정보(양자화한 텐서 목록) |
| `models/qwen/int8.safetensors` | 2,162,922,376 | `61e3adb3` | C v4 제출(03 `minprompt`)의 Qwen3-VL-Reranker-2B + LoRA, int8 |

설정·토크나이저 파일(각 모델 원본 그대로)
| zip 안 경로 | 바이트 | CRC32 |
|---|---|---|
| `models/siglip/config.json` | 995 | `2716a2d6` |
| `models/siglip/processor_config.json` | 463 | `b9d8870c` |
| `models/siglip/spiece.model` | 798,330 | `b21a34dd` |
| `models/siglip/tokenizer_config.json` | 822 | `b86b9e67` |
| `models/blip/config.json` | 1,940 | `03afe427` |
| `models/blip/processor_config.json` | 529 | `9912abf7` |
| `models/blip/tokenizer.json` | 712,013 | `adff099f` |
| `models/blip/tokenizer_config.json` | 696 | `13fd0d96` |
| `models/qwen/config.json` | 1,754 | `640852fc` |
| `models/qwen/processor_config.json` | 1,286 | `d8cfc38a` |
| `models/qwen/tokenizer.json` | 11,422,817 | `bf57e015` |
| `models/qwen/tokenizer_config.json` | 837 | `0e1487be` |
| `models/qwen/chat_template.jinja` | 5,292 | `54a6b02d` |
| `models/qwen/additional_chat_templates/reranker.jinja` | 2,443 | `2b28dd1e` |

제출별로 달라지는 두 파일
| 제출 | `model.py` | `models/ens_config.json` |
|---|---|---|
| 11 | 20,987바이트, CRC32 `7f5c404c` (= `model_zip_min.py`) | 114바이트, CRC32 `e704fd47` (= `ens_config.k20_qwen2_fuse.json` + 뒤 공백) |

## BLIP LoRA (추가 학습분)
- `train/blip_ft.py`로 만든 `blip_lora_b.safetensors` (9,717,540바이트, md5 `b78a714004a153020a6e968e2e324d88`): Q-Former Linear 120개의 LoRA(r16, alpha32) + `itm_head`.
- 위 `models/blip/model_q8.safetensors`를 되돌린 가중치 위에서 학습했으므로 **같은 int8 파일과 함께** 써야 한다. zip에는 `models/blip/lora_ft.safetensors`로 넣는다.
- 10MB라 git에 올릴 수 있다(저장소가 공개라 올리는 시점은 팀이 정함).
