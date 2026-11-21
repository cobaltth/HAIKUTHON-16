"""모델 어댑터: 백본마다 달라지는 부분(불러오기·전처리·임베딩·학습 층·손실 종류)만 여기에 모은다.
새 백본(예: A의 베이스라인)을 쓰려면 같은 메서드를 가진 클래스를 하나 추가하고 ADAPTERS에 등록하면 된다."""
import torch
import torch.nn.functional as F


def _pooled(out):
    return out.pooler_output if hasattr(out, "pooler_output") else out


class SiglipAdapter:
    loss_type = "sigmoid"

    def __init__(self, model_name, device):
        from transformers import AutoProcessor, SiglipModel
        self.net = SiglipModel.from_pretrained(model_name).to(device)
        self.proc = AutoProcessor.from_pretrained(model_name)
        self.device = device

    # --- 학습할 층 고르기 -------------------------------------------------
    def set_trainable(self, mode, train_logit=False):
        """mode: heads(스타터킷과 동일) | lastK (예: last2 = 두 인코더의 마지막 K개 층 + 헤드 + 최종 LayerNorm) | all"""
        n = self.net
        n.requires_grad_(False)
        tm, vm = n.text_model, n.vision_model
        heads = [tm.head] + ([vm.head] if hasattr(vm, "head") else [])
        for m in heads:
            m.requires_grad_(True)
        if mode.startswith("last"):
            k = int(mode[4:])
            for m in list(tm.encoder.layers[-k:]) + list(vm.encoder.layers[-k:]) + [tm.final_layer_norm, vm.post_layernorm]:
                m.requires_grad_(True)
        elif mode == "all":
            n.requires_grad_(True)
        elif mode != "heads":
            raise ValueError(mode)
        n.logit_scale.requires_grad_(train_logit)
        n.logit_bias.requires_grad_(train_logit)
        return sum(p.numel() for p in n.parameters() if p.requires_grad)

    # --- 전처리 / 임베딩 ---------------------------------------------------
    def pixels(self, images):
        return self.proc(images=images, return_tensors="pt").pixel_values.to(self.device)

    def tokens(self, texts):
        # 스타터킷 model.py와 같은 방식(64토큰 max_length 패딩)으로 맞춰야 학습=추론 조건이 같다
        return self.proc(text=texts, padding="max_length", truncation=True, return_tensors="pt").to(self.device)

    def encode_image(self, pixels):
        return F.normalize(_pooled(self.net.get_image_features(pixel_values=pixels)).float(), dim=-1)

    def encode_text(self, tokens):
        return F.normalize(_pooled(self.net.get_text_features(**tokens)).float(), dim=-1)

    def loss_params(self):
        return {"logit_scale": self.net.logit_scale, "logit_bias": self.net.logit_bias}

    # --- 추론/저장 (스타터킷 model.py 형식) ---------------------------------
    def as_submission_model(self):
        return (self.net, self.proc)

    def save(self, folder):
        self.net.save_pretrained(folder)
        self.proc.save_pretrained(folder)


ADAPTERS = {"siglip": SiglipAdapter}


def get_adapter(name, model_name, device):
    return ADAPTERS[name](model_name, device)
