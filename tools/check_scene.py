"""scene Qwen int8.safetensors가 zip 안의 v4 int8.safetensors와 같은 구조인지 확인 (텐서 이름·dtype·shape·파일 크기).
 사용: python check_scene.py <zip> <새 int8.safetensors>"""
import json, os, struct, sys, zipfile

zp, new = sys.argv[1:3]
name = "models/qwen/int8.safetensors"
z = zipfile.ZipFile(zp)
with z.open(name) as f:
    n = struct.unpack("<Q", f.read(8))[0]; h_old = json.loads(f.read(n))
with open(new, "rb") as f:
    n2 = struct.unpack("<Q", f.read(8))[0]; h_new = json.loads(f.read(n2))
ko = {k: (v["dtype"], tuple(v["shape"])) for k, v in h_old.items() if k != "__metadata__"}
kn = {k: (v["dtype"], tuple(v["shape"])) for k, v in h_new.items() if k != "__metadata__"}
miss, extra = sorted(set(ko) - set(kn)), sorted(set(kn) - set(ko))
diff = [k for k in ko if k in kn and ko[k] != kn[k]]
so, sn = z.getinfo(name).file_size, os.path.getsize(new)
print("tensors old/new:", len(ko), len(kn), "| missing in new:", len(miss), miss[:5], "| extra in new:", len(extra), extra[:5])
print("dtype/shape differences:", len(diff), diff[:5])
print("header bytes old/new:", n, n2, "| file bytes old/new:", so, sn, "| same size:", so == sn)
print("metadata old:", str(h_old.get("__metadata__"))[:300])
print("metadata new:", str(h_new.get("__metadata__"))[:300])
dt = {}
for k, (d, s) in kn.items():
    dt[d] = dt.get(d, 0) + 1
print("dtype counts new:", dt)
print("RESULT:", "COMPATIBLE" if not miss and not extra and not diff and so == sn else ("SAME_TENSORS_DIFFERENT_SIZE" if not miss and not extra and not diff else "DIFFERENT"))
