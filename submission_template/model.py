"""평가 서버가 불러오는 SigLIP 검색 모델."""

from pathlib import Path

import torch
from PIL import Image
from torch.nn.functional import normalize
from transformers import AutoProcessor, SiglipModel


def load_model(model_dir, device):
    # 서버가 전달한 ZIP 최상위 경로에서 이 예제의 models/ 폴더를 찾습니다.
    checkpoint_dir = Path(model_dir) / "models"
    network = SiglipModel.from_pretrained(checkpoint_dir, local_files_only=True).to(device)
    if network.device.type == "cuda":
        network.half()
    processor = AutoProcessor.from_pretrained(checkpoint_dir, local_files_only=True)
    return network.eval(), processor


def read_image(path):
    with Image.open(path) as image:
        return image.convert("RGB")


def tokenize(processor, prompts):
    # SigLIP 사전 학습 때와 같은 길이로 패딩합니다.
    return processor(text=prompts, padding="max_length", truncation=True, return_tensors="pt")


@torch.inference_mode()
def predict(model, queries, candidates, top_k=5):
    network, processor = model
    network.eval()

    # 모든 프롬프트가 같은 후보 풀을 사용하므로 이미지당 한 번만 인코딩합니다.
    image_batches = []
    for start in range(0, len(candidates), 32):
        batch = candidates[start:start + 32]
        images = [read_image(item["image_path"]) for item in batch]
        pixels = processor(images=images, return_tensors="pt").pixel_values
        pixels = pixels.to(device=network.device, dtype=network.dtype)
        features = network.get_image_features(pixel_values=pixels).pooler_output
        image_batches.append(normalize(features.float(), dim=-1))
    image_vectors = torch.cat(image_batches)
    image_ids = [item["image_id"] for item in candidates]

    # 각 행은 프롬프트 하나의 코사인 유사도입니다. 행마다 독립적으로 순위를 구합니다.
    predictions = {}
    for start in range(0, len(queries), 128):
        batch = queries[start:start + 128]
        tokens = tokenize(processor, [item["prompt"] for item in batch]).to(network.device)
        features = network.get_text_features(**tokens).pooler_output
        scores = normalize(features.float(), dim=-1) @ image_vectors.T
        indices = scores.topk(top_k, dim=1).indices.tolist()
        for query, row in zip(batch, indices):
            predictions[query["prompt_id"]] = [image_ids[index] for index in row]
    return predictions
