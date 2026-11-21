"""학습용 샘플 생성 + 그룹 배치 샘플러.

make_pair(row, rng, cfg) -> (PIL 이미지, 캡션)   : 정답 쌍 (증강 적용)
GroupBatchSampler(ids, groups, batch_size)       : 형제 캡션이 같은 배치에 들어가도록 id 배치를 생성
"""
import random, csv, collections
from PIL import Image
from .text_aug import swap_left_right, has_left_right, hard_negatives
from .image_aug import weak_square_crop, hflip

DEFAULT_CFG = {"flip_swap_p": 0.0, "crop_p": 0.0, "n_text_neg": 0}  # 기본은 전부 끔 → 하나씩 켜며 ablation


def load_groups(path):
    with open(path, newline="") as f:
        return {r["id"]: int(r["group_id"]) for r in csv.DictReader(f)}


def make_pair(image_path, caption, rng, cfg=DEFAULT_CFG):
    with Image.open(image_path) as im:
        img = im.convert("RGB")
    if cfg.get("crop_p", 0) and rng.random() < cfg["crop_p"]:
        img = weak_square_crop(img, rng)
    flipped = False
    if cfg.get("flip_swap_p", 0) and rng.random() < cfg["flip_swap_p"]:
        img, caption, flipped = hflip(img), swap_left_right(caption), True
    negs = []
    if cfg.get("n_text_neg", 0):
        cands = hard_negatives(caption, rng)
        negs = [c for _, c in rng.sample(cands, min(cfg["n_text_neg"], len(cands)))]
    return img, caption, negs


class GroupBatchSampler:
    """그룹(형제)을 통째로 같은 배치에 넣는다. 그룹이 배치보다 크면 쪼개서 연속 배치로.
    singleton은 빈자리 채우기용. 매 에포크 그룹 순서를 섞음."""

    def __init__(self, ids, groups, batch_size, seed=0, drop_last=False):
        self.batch_size, self.seed, self.drop_last, self.epoch = batch_size, seed, drop_last, 0
        by_g = collections.defaultdict(list)
        for i in ids:
            by_g[groups[i]].append(i)
        self.groups = list(by_g.values())

    def set_epoch(self, e):
        self.epoch = e

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        gs = [g[:] for g in self.groups]
        rng.shuffle(gs)
        for g in gs: rng.shuffle(g)
        multi = [g for g in gs if len(g) > 1]
        single = [g[0] for g in gs if len(g) == 1]
        batch = []
        for g in multi:
            for i in g:
                batch.append(i)
                if len(batch) == self.batch_size:
                    yield batch; batch = []
            # 남은 자리가 적으면 singleton으로 채워 다음 그룹이 잘리지 않게
            space = self.batch_size - len(batch)
            if batch and 0 < space < 3 and single:
                while len(batch) < self.batch_size and single:
                    batch.append(single.pop())
                yield batch; batch = []
        for i in single:
            batch.append(i)
            if len(batch) == self.batch_size:
                yield batch; batch = []
        if batch and not self.drop_last:
            yield batch

    def __len__(self):
        n = sum(len(g) for g in self.groups)
        return n // self.batch_size + (0 if self.drop_last else 1)
