"""이미지 증강: 약한 정사각 크롭, 좌우 반전. 색 변형은 금지(색이 정답 속성)."""
import random
from PIL import Image, ImageOps


def weak_square_crop(img, rng=None, min_scale=0.9):
    """한 변을 원본의 min_scale~1.0배로 (기본 0.9: 0.8에서는 가장자리 물체가 반쯤 잘리는 사례 확인) 자른 뒤 원래 크기로 되돌림. 물체가 잘려 개수가 바뀌지 않도록 약하게만."""
    rng = rng or random
    w, h = img.size
    s = rng.uniform(min_scale, 1.0)
    cw, ch = int(w * s), int(h * s)
    x, y = rng.randint(0, w - cw), rng.randint(0, h - ch)
    return img.crop((x, y, x + cw, y + ch)).resize((w, h), Image.BICUBIC)


def hflip(img):
    return ImageOps.mirror(img)
