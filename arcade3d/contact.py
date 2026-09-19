"""Tile the rendered stills into one labeled contact sheet."""
import sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

src = Path(sys.argv[1] if len(sys.argv) > 1 else "stills")
out = Path(sys.argv[2] if len(sys.argv) > 2 else "contact.png")
cols = int(sys.argv[3]) if len(sys.argv) > 3 else 5
files = sorted(src.glob("t*.png"))
tw, th = 360, 450
rows = (len(files) + cols - 1) // cols
sheet = Image.new("RGB", (cols * tw, rows * th), "black")
font = ImageFont.truetype("arial.ttf", 22)
for i, f in enumerate(files):
    im = Image.open(f).convert("RGB").resize((tw, th), Image.LANCZOS)
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, 90, 30], fill="black")
    d.text((6, 3), f.stem[1:] + "s", fill="white", font=font)
    sheet.paste(im, ((i % cols) * tw, (i // cols) * th))
sheet.save(out)
print(out, sheet.size)
