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
| 파일 | 바이트 | md5 | 학습 | 쓴 제출 |
|---|---|---|---|---|
| `blip_lora_b.safetensors` | 9,717,540 | `b78a714004a153020a6e968e2e324d88` | `train/blip_ft.py`: 8장 세트 276개, lr 6e-4, 4바퀴째 | 13 |
| `blip_lora_c.safetensors` | 9,717,540 | `1fbdfbe24d92a209c9c7bc9a7920f73f` | `train/blip_ft2.py`: 8장 세트 276개 + 4장 세트 339개, lr 6e-4, 4바퀴째 | 15 |

- 내용: Q-Former Linear 120개의 LoRA(r16, alpha32) + `itm_head`. zip에는 `models/blip/lora_ft.safetensors`로 넣는다.
- 위 `models/blip/model_q8.safetensors`를 되돌린 가중치 위에서 학습했으므로 **같은 int8 파일과 함께** 써야 한다.
- 10MB라 git에 올릴 수 있다(저장소가 공개라 올리는 시점은 팀이 정함).

## Qwen scene (제출 15)
- zip 안 `models/qwen/int8.safetensors`를 C의 Drive `haikuthon/qwen_lora_scene/int8.safetensors`로 제자리 교체했다.
- 크기 2,162,922,376바이트(v4와 같음), CRC32 `968a464c` (v4는 `61e3adb3`). 텐서 927개의 이름·dtype·shape가 v4와 같다. 프롬프트는 v4와 같은 빈 프롬프트.
- 제출 15의 zip: `sub_final_c_scene.zip` (3,886,200,008바이트). `model.py`·설정·LoRA는 `zip_append.py`로 덧붙였기 때문에, 목차에서 빠진 옛 항목의 바이트가 본문에 조금 남아 있다(무시됨).

## 내지 않은 예비 zip (LoRA c + Qwen scene, K28·상위 2장)
- `sub_final2_c_scene_k28_qwen2.zip` (3,886,200,216바이트, 항목 22개). 15번을 올린 뒤 같은 zip에 `models/ens_config.json`만 `zip_append.py`로 덧붙이고 이름을 바꾼 것이라, 15번 zip보다 208바이트 크고 나머지는 같다.
- 15번 zip을 다시 만들려면 이 파일의 설정만 `ens_config.lora_c_scene_k20_qwen3.json`으로 되돌리면 된다.

| zip 안 경로 | 바이트 | CRC32 | 내용 |
|---|---|---|---|
| `model.py` | 28,750 | `e5391846` | `submission_ens/model.py` (v5, md5 c2d80207…) |
| `models/ens_config.json` | 156 | `48e085c9` | `ens_config.lora_c_scene_k28_qwen2.json` |
| `models/blip/lora_ft.safetensors` | 9,717,540 | `fc67db23` | `blip_lora_c.safetensors` |
| `models/qwen/int8.safetensors` | 2,162,922,376 | `968a464c` | Qwen scene |
| `models/blip/model_q8.safetensors` | 1,284,342,816 | `a9388d01` | 위 표와 같음 |
| `models/siglip/model.safetensors` | 406,360,284 | `7f62f2da` | 위 표와 같음 |
