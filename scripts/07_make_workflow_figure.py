"""Create the study-design flowchart from the shoreline study workflow."""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "extraction" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# The wide layout remains legible when reduced to the journal's single-column
# figure width.  Labels use a serif face and a color-blind-friendly palette.
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 14,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    }
)
COLORS = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "purple": "#CC79A7",
    "teal": "#009E73",
    "sky": "#56B4E9",
    "ink": "#203044",
    "pale_blue": "#E5F4FB",
    "pale_orange": "#FFF5DD",
    "pale_purple": "#F8EAF2",
    "pale_teal": "#E7F5EF",
}

fig, ax = plt.subplots(figsize=(13.0, 6.35))
ax.set_xlim(0, 13)
ax.set_ylim(0, 6.35)
ax.axis("off")


def box(x, y, w, h, title, body, edge, face, title_size=16, body_size=13.5):
    shape = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=.06,rounding_size=.12",
        facecolor=face,
        edgecolor=edge,
        linewidth=1.8,
    )
    ax.add_patch(shape)
    ax.text(
        x + w / 2,
        y + h * 0.68,
        title,
        ha="center",
        va="center",
        fontsize=title_size,
        fontweight="bold",
        color=COLORS["ink"],
    )
    ax.text(
        x + w / 2,
        y + h * 0.32,
        body,
        ha="center",
        va="center",
        fontsize=body_size,
        color=COLORS["ink"],
        linespacing=1.16,
    )


def arrow(start, end, color=COLORS["ink"], rad=0):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=15,
            linewidth=1.8,
            color=color,
            connectionstyle=f"arc3,rad={rad}",
        )
    )


# Input acquisition and index calculation.
box(
    0.45,
    4.85,
    4.4,
    1.0,
    "Sentinel-2 Level-2A",
    "32 seasonal composites • 2018–2025",
    COLORS["sky"],
    COLORS["pale_blue"],
)
box(
    8.15,
    4.85,
    4.4,
    1.0,
    "MNDWI rasters",
    "(Green − SWIR1) / (Green + SWIR1)",
    COLORS["sky"],
    COLORS["pale_blue"],
)
arrow((4.95, 5.35), (8.0, 5.35), COLORS["sky"])

# Competing water-mask approaches share the same vectorisation chain.
box(
    0.45,
    3.15,
    5.3,
    1.15,
    "Classical MNDWI pipeline",
    "Thresholding and mask cleaning\nReference water masks",
    COLORS["orange"],
    COLORS["pale_orange"],
)
box(
    7.25,
    3.15,
    5.3,
    1.15,
    "CNN segmentation",
    "U-Net • SegNet • FCN\n128 × 128 patches; reference-mask supervision",
    COLORS["purple"],
    COLORS["pale_purple"],
)
arrow((9.05, 4.8), (4.8, 4.37), COLORS["orange"], rad=-0.08)
arrow((10.95, 4.8), (9.9, 4.37), COLORS["purple"], rad=0.08)

box(
    3.45,
    1.75,
    6.1,
    0.92,
    "Shared shoreline vectorisation",
    "Mask threshold • cleanup • contour filtering • UTM zone 29N",
    COLORS["teal"],
    COLORS["pale_teal"],
    title_size=15.5,
    body_size=13.2,
)
arrow((3.1, 3.1), (5.05, 2.72), COLORS["orange"], rad=0.02)
arrow((9.9, 3.1), (7.95, 2.72), COLORS["purple"], rad=-0.02)

# Three evaluation levels use the same derived shoreline set.
outputs = [
    (0.3, "Pixel agreement", "IoU / F1 vs.\nMNDWI-derived labels", COLORS["blue"]),
    (4.55, "Vector geometry", "Shoreline length /\nmean positional offset", COLORS["orange"]),
    (8.8, "Coastal change", "EPR / LRR / NSM\n40 common transects", COLORS["purple"]),
]
for x, title, body, color in outputs:
    box(x, 0.2, 3.9, 0.95, title, body, color, "white", title_size=14.5, body_size=12.8)
    arrow((6.5, 1.7), (x + 1.95, 1.18), COLORS["teal"], rad=(x - 4.55) * 0.025)

fig.subplots_adjust(left=0.015, right=0.985, top=0.98, bottom=0.035)
for suffix, kwargs in (
    ("png", {"dpi": 320}),
    ("pdf", {}),
):
    fig.savefig(OUT / f"Fig12_experimental_design.{suffix}", bbox_inches="tight", pad_inches=0.06, **kwargs)
plt.close(fig)
print(OUT / "Fig12_experimental_design.png")
