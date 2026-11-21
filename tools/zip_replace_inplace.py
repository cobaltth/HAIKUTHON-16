"""제출 zip 안의 큰 파일(무압축 저장)을 '같은 크기'의 새 파일로 제자리에서 바꾼다.
 zip_append.py로 바꾸면 옛 파일의 바이트가 zip 본문에 남아 용량이 그만큼 커지므로(5GB 제한), 가중치처럼 큰 파일은 이 도구를 쓴다.
 하는 일: 데이터 영역 덮어쓰기 → 지역 헤더와 목차의 CRC32 갱신 → 다시 읽어 검증.
 사용: python zip_replace_inplace.py <zip> <zip 안 경로> <새 파일>
"""
import os, struct, sys, time, zipfile, zlib


def main():
    zp, name, newfile = sys.argv[1:4]
    with zipfile.ZipFile(zp) as z:
        zi = z.getinfo(name)
        start_dir = z.start_dir
        assert zi.compress_type == zipfile.ZIP_STORED, "entry is compressed"
        size, hoff, old_crc = zi.file_size, zi.header_offset, zi.CRC
    nsize = os.path.getsize(newfile)
    if nsize != size:
        raise SystemExit(f"size differs: zip entry {size} vs new file {nsize} -> in-place replace impossible")
    t0 = time.time()
    with open(zp, "r+b") as f:
        f.seek(hoff)
        sig, ver, flag, comp, mt, md, crc, csz, usz, n, m = struct.unpack("<IHHHHHIIIHH", f.read(30))
        assert sig == 0x04034B50 and comp == 0 and not (flag & 8), "unexpected local header"
        assert f.read(n).decode("utf-8", "replace") == name, "local header name mismatch"
        data_off = hoff + 30 + n + m
        assert data_off + size <= start_dir
        crc_new = 0
        f.seek(data_off)
        done = 0
        with open(newfile, "rb") as g:
            while True:
                b = g.read(16 << 20)
                if not b:
                    break
                crc_new = zlib.crc32(b, crc_new)
                f.write(b)
                done += len(b)
                if done % (256 << 20) < (16 << 20):
                    print(f"  {done/1e9:.2f} / {size/1e9:.2f} GB  {time.time()-t0:.0f}s", flush=True)
        assert done == size
        crc_new &= 0xFFFFFFFF
        f.seek(hoff + 14); f.write(struct.pack("<I", crc_new))
        f.seek(start_dir)
        pos = start_dir
        while True:
            rec = f.read(46)
            if len(rec) < 46 or rec[:4] != b"PK\x01\x02":
                raise SystemExit("central directory record not found")
            n2, m2, k2 = struct.unpack("<HHH", rec[28:34])
            fn = f.read(n2).decode("utf-8", "replace")
            if fn == name:
                f.seek(pos + 16); f.write(struct.pack("<I", crc_new))
                break
            f.seek(m2 + k2, 1)
            pos += 46 + n2 + m2 + k2
        f.flush(); os.fsync(f.fileno())
    print(f"written {size} bytes in {time.time()-t0:.0f}s, crc {old_crc:08x} -> {crc_new:08x}", flush=True)
    with zipfile.ZipFile(zp) as z:      # 검증: zipfile이 끝까지 읽으며 CRC를 확인한다
        c = 0
        with z.open(name) as f:
            while True:
                b = f.read(16 << 20)
                if not b:
                    break
                c = zlib.crc32(b, c)
        assert (c & 0xFFFFFFFF) == crc_new == z.getinfo(name).CRC
        names = z.namelist()
        assert len(names) == len(set(names))
    print(f"verified ok ({time.time()-t0:.0f}s total); zip size {os.path.getsize(zp)}", flush=True)


if __name__ == "__main__":
    main()
