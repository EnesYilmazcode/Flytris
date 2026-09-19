"""Animate a cinematic push from the generated fly arcade shot into a replay frame."""
import argparse
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image

W, H = 1080, 1350


def ease(value):
    return value * value * (3.0 - 2.0 * value)


def cover(image):
    scale = max(W / image.width, H / image.height)
    size = (round(image.width * scale), round(image.height * scale))
    image = image.resize(size, Image.Resampling.LANCZOS)
    left = (image.width - W) // 2
    top = (image.height - H) // 2
    return image.crop((left, top, left + W, top + H))


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--hold", type=float, default=1.7)
    parser.add_argument("--push", type=float, default=2.0)
    args = parser.parse_args(argv)

    scene = cover(Image.open(args.image).convert("RGB"))
    target = cover(Image.open(args.target).convert("RGB"))
    # Bounding box of the generated cabinet's screen after the near-identity cover crop.
    screen = np.array([405.0, 226.0, 817.0, 742.0])
    full = np.array([0.0, 0.0, float(W), float(H)])
    hold_frames = round(args.hold * args.fps)
    push_frames = round(args.push * args.fps)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    ff = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{W}x{H}", "-r", str(args.fps), "-i", "-", "-c:v", "libx264",
         "-pix_fmt", "yuv420p", "-crf", "18", str(args.out)],
        stdin=subprocess.PIPE,
    )
    for frame in range(hold_frames + push_frames):
        if frame < hold_frames:
            image = scene
        else:
            raw = (frame - hold_frames + 1) / push_frames
            amount = ease(min(1.0, raw))
            box = full + (screen - full) * amount
            image = scene.crop(tuple(box)).resize((W, H), Image.Resampling.LANCZOS)
            crossfade = max(0.0, (raw - 0.72) / 0.28)
            if crossfade:
                image = Image.blend(image, target, ease(min(1.0, crossfade)))
        ff.stdin.write(np.asarray(image).tobytes())
    ff.stdin.close()
    if ff.wait() != 0:
        raise RuntimeError("ffmpeg failed")
    print(f"rendered {args.out}")


if __name__ == "__main__":
    main()
