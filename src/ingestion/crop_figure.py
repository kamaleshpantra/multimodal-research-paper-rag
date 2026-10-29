from pathlib import Path
import sys
import pymupdf

if len(sys.argv) < 4:
    print('Usage: python crop_figure.py PDF PAGE "CAPTION TEXT" [OUTPUT]')
    raise SystemExit(2)

pdf_path = Path(sys.argv[1])
page_number = int(sys.argv[2])
caption = sys.argv[3]

output = Path(
    sys.argv[4] if len(sys.argv) > 4
    else "data/images/figure_crop.png"
)
output.parent.mkdir(parents=True, exist_ok=True)

with pymupdf.open(pdf_path) as doc:
    if not 1 <= page_number <= len(doc):
        raise ValueError(f"PDF has {len(doc)} pages.")

    page = doc[page_number - 1]
    matches = page.search_for(caption)

    if not matches:
        print("Caption not found. Available page text:")
        print(page.get_text("text")[:2500])
        raise SystemExit(1)

    # Figure captions usually sit below the figure.
    caption_box = matches[0]
    clip = pymupdf.Rect(
        0,
        max(0, caption_box.y0 - 480),
        page.rect.width,
        min(page.rect.height, caption_box.y1 + 20),
    )

    pix = page.get_pixmap(
        matrix=pymupdf.Matrix(2, 2),
        clip=clip,
        alpha=False,
    )
    pix.save(output)

print("Saved figure crop:", output.resolve())
