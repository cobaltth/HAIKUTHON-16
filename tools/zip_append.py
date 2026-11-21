"""제출 zip에 파일을 '제자리에서' 추가/교체한다 (3.9GB 전체를 다시 쓰지 않음).
 새 항목을 기존 목차 자리에 이어 쓰고 목차를 다시 쓴다. 교체되는 옛 항목은 목차에서만 빠진다(본문 바이트는 남지만 무시됨).
 사용: python zip_append.py <zip> <zip 안 경로>=<파일 경로> ... [--drop <zip 안 경로> ...]
"""
import sys, zipfile, time


def main():
    zp = sys.argv[1]
    adds, drops, mode = [], [], "add"
    for a in sys.argv[2:]:
        if a == "--drop":
            mode = "drop"; continue
        if mode == "drop":
            drops.append(a)
        else:
            k, v = a.split("=", 1); adds.append((k, v))
    with zipfile.ZipFile(zp, "a", zipfile.ZIP_STORED) as z:
        for name in drops + [k for k, _ in adds]:
            if name in z.NameToInfo:
                z.filelist.remove(z.NameToInfo.pop(name))
        for k, v in adds:
            zi = zipfile.ZipInfo(k, date_time=time.localtime()[:6])
            zi.compress_type = zipfile.ZIP_STORED
            zi.external_attr = 0o644 << 16
            z.writestr(zi, open(v, "rb").read())
        z._didModify = True
    with zipfile.ZipFile(zp) as z:
        bad = z.testzip() if "--full" in sys.argv else None
        names = z.namelist()
        assert len(names) == len(set(names)), "duplicate names"
        for k, v in adds:
            assert z.read(k) == open(v, "rb").read(), k
        print("ok", len(names), "entries; total", sum(i.file_size for i in z.infolist()), "bytes;", bad)
        for i in z.infolist():
            if i.filename in dict(adds) or i.filename.endswith(".json") and "ens_config" in i.filename:
                print("  ", i.filename, i.file_size, i.header_offset)


if __name__ == "__main__":
    main()
