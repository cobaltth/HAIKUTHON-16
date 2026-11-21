"""SigLIP 손실 + 텍스트 hard negative 추가 버전.

기본 SigLIP: 배치 B쌍의 이미지·텍스트 점수 B×B 행렬에서 대각선=정답(+1), 나머지=오답(-1)로 시그모이드 손실.
여기에 이미지마다 '자기 캡션을 살짝 바꾼 가짜 캡션'(neg_txt)을 열로 덧붙여 오답(-1)으로 학습한다.
"""
import torch
import torch.nn.functional as F


def siglip_loss(img, txt, logit_scale, logit_bias, neg_txt=None, neg_owner=None, neg_weight=1.0):
    """img: (B,D), txt: (B,D) 정규화된 임베딩. neg_txt: (N,D), neg_owner: (N,) 각 가짜 캡션의 원래 이미지 인덱스.
    logit_scale/logit_bias: 모델의 학습 파라미터(model.logit_scale, model.logit_bias)."""
    scale = logit_scale.exp()
    logits = img @ txt.T * scale + logit_bias                        # (B,B)
    labels = 2 * torch.eye(len(img), device=img.device) - 1          # 대각 +1, 나머지 -1
    loss = -F.logsigmoid(labels * logits).sum() / len(img)
    if neg_txt is not None and len(neg_txt):
        # 가짜 캡션은 자기 원래 이미지에 대해서만 오답으로 사용 (다른 이미지와는 원래부터 무관)
        neg_logits = (img[neg_owner] * neg_txt).sum(-1) * scale + logit_bias
        loss = loss + neg_weight * (-F.logsigmoid(-neg_logits)).sum() / len(img)
    return loss


def clip_loss(img, txt, logit_scale, neg_txt=None, neg_owner=None):
    """CLIP 계열(소프트맥스) 손실. 가짜 캡션은 이미지→텍스트 방향의 추가 오답 열로 붙인다.
    (가짜 캡션은 원래 이미지에게는 '거의 맞지만 틀린' 오답, 다른 이미지에게는 무관한 오답이라 전부 오답으로 써도 됨)"""
    scale = logit_scale.exp()
    logits = img @ txt.T * scale
    target = torch.arange(len(img), device=img.device)
    i2t_logits = logits if neg_txt is None or not len(neg_txt) else torch.cat([logits, img @ neg_txt.T * scale], 1)
    return (F.cross_entropy(i2t_logits, target) + F.cross_entropy(logits.T, target)) / 2
