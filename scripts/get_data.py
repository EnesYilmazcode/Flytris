"""Download MaleCNS v1.0 (1.1 GB, CC BY 4.0) and convert it for Flytris.

Resumes partial downloads, checks sha256, renames into place only when complete, then
runs data/malecns/build.py and builds the photoreceptor map.

    python -X utf8 scripts/get_data.py
"""
import hashlib
import os
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/malecns/raw"
BASE = "https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/"
FILES = {
    "body-annotations-male-cns-v1.0-minconf-0.5.feather":
        (14_483_314, "2177e246113e4cfbf1e7772ec37c6da1955ff22e8063d0b1f833101f99a9a3b2"),
    "body-neurotransmitters-male-cns-v1.0.feather":
        (43_282_834, "95c9289220663abeb3409f3ad9e5a7f8a53f8093f5139d15502cd08da8879621"),
    "connectome-weights-male-cns-v1.0-minconf-0.5.feather":
        (1_051_241_946, "e35da783d1c686b2b58b3b87cd6a403ae43bfcfba8bff28e08ef752c1a56afc1"),
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(name, size, digest):
    final, part = RAW / name, RAW / (name + ".part")
    if final.exists() and final.stat().st_size == size and sha256(final) == digest:
        print(f"ok       {name}")
        return
    have = part.stat().st_size if part.exists() else 0
    if have < size:
        req = urllib.request.Request(BASE + name, headers={"Range": f"bytes={have}-"})
        with urllib.request.urlopen(req) as r, open(part, "ab" if have else "wb") as f:
            if have and r.status != 206:
                f.seek(0)
                f.truncate()
                have = 0
            done = have
            while chunk := r.read(1 << 22):
                f.write(chunk)
                done += len(chunk)
                print(f"\r{name}: {done / size:6.1%}", end="", flush=True)
        print()
    if sha256(part) != digest:
        part.unlink()
        sys.exit(f"sha256 mismatch for {name}; deleted the partial file, run again")
    os.replace(part, final)
    print(f"verified {name}")


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    for name, (size, digest) in FILES.items():
        fetch(name, size, digest)
    subprocess.run([sys.executable, "-X", "utf8", str(ROOT / "data/malecns/build.py")], check=True)
    sys.path.insert(0, str(ROOT))
    from flytris.eyes import MAP, build_eye_map
    if MAP.exists():
        MAP.unlink()
    build_eye_map()
    print("done. Next: python -X utf8 scripts/decode_check.py --save runs/readout_groups.npz")


if __name__ == "__main__":
    main()
