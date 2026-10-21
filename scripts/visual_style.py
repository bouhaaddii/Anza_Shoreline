"""Shared, accessible visual theme for the shoreline study figures."""

from matplotlib.colors import LinearSegmentedColormap


# A restrained categorical palette used consistently for each extraction method.
MODEL_COLORS = {
    "classical": "#465362",  # slate
    "unet": "#12A6A6",       # teal
    "segnet": "#F5A431",     # amber
    "fcn": "#D94F64",        # rose
}

# The threshold at zero is meaningful in both indices: negative and positive
# values should therefore diverge from a quiet, near-neutral midpoint.
MNDWI_CMAP = LinearSegmentedColormap.from_list(
    "shoreline_mndwi", ["#C85A4A", "#F7F5F2", "#2878A5"], N=256
)
CHANGE_CMAP = LinearSegmentedColormap.from_list(
    "shoreline_change", ["#C85A4A", "#F7F5F2", "#2878A5"], N=256
)
PROBABILITY_CMAP = LinearSegmentedColormap.from_list(
    "shoreline_probability", ["#F3F5F6", "#168A9A"], N=256
)

EROSION = "#C85A4A"
ACCRETION = "#2878A5"
STABLE = "#A8ADB3"
REFERENCE = "#F28E2B"

STYLE = {
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "axes.linewidth": 0.8,
    "axes.edgecolor": "#89939E",
    "axes.labelcolor": "#26323D",
    "axes.titlecolor": "#1F2933",
    "axes.titleweight": "bold",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.axisbelow": True,
    "xtick.color": "#465362",
    "ytick.color": "#465362",
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "legend.frameon": False,
    "figure.dpi": 150,
    "figure.titlesize": 13,
    "figure.titleweight": "bold",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.dpi": 320,
    "savefig.bbox": "tight",
    "savefig.facecolor": "white",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "grid.color": "#D9E0E5",
    "grid.linewidth": 0.6,
    "lines.solid_capstyle": "round",
}
