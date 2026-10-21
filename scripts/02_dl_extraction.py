#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
STUDY 1 - STEP 02: DEEP-LEARNING SHORELINE EXTRACTION (U-Net / SegNet / FCN)

Detailed, fully documented implementation of the deep-learning pipeline:

  1. Training labels  : cleaned water masks of step 01 (weak supervision).
  2. Patch extraction : 128 x 128 patches, 64-px stride (50% overlap),
                        stratified 70 / 15 / 15 train / val / test split.
  3. Architectures    : U-Net (skip connections), SegNet (pooling indices
                        surrogate), FCN (progressive up-sampling) - single
                        MNDWI channel in, sigmoid water probability out.
  4. Optimisation     : Adam (lr 1e-4), hybrid loss 0.5*BCE + 0.5*(1 - Dice),
                        EarlyStopping (patience 10), ReduceLROnPlateau,
                        up to 50 epochs, batch 16.
  5. Inference        : sliding-window full-raster prediction (averaged
                        overlaps), 0.5 binarisation.
  6. Vectorisation    : STRICTLY the same chain as the classical pipeline
                        (extraction_core), guaranteeing a fair comparison.

Set RETRAIN = False to reuse existing checkpoints (outputs/dl/ckpts/ or the
local models/checkpoints) instead of training from scratch.

Outputs: outputs/dl_shorelines/                per-model shapefiles + CSV
         outputs/dl/ckpts/*.keras              trained models
         outputs/dl/history_*.json             training histories
         outputs/tables/pixel_metrics.csv      test-set IoU/F1/Acc/Prec/Rec
         outputs/figures/Fig08_training_curves.(png|pdf)

Run:  python scripts/02_dl_extraction.py
"""

import json
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import extraction_core as core

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "data"
OUT = BASE / "outputs" / "extraction"
LABEL_DIR = OUT / "classical_shorelines" / "masks"    # produced by step 01
LABEL_FALLBACK = DATA / "labels"                       # pre-computed labels
CKPT_DIR = OUT / "dl" / "ckpts"
CKPT_FALLBACK = BASE / "models" / "checkpoints"
SHORE_DIR = OUT / "dl_shorelines"
FIG = OUT / "figures"
TAB = OUT / "tables"
for d in (CKPT_DIR, SHORE_DIR, FIG, TAB, OUT / "dl"):
    d.mkdir(parents=True, exist_ok=True)

RETRAIN = False          # True: always train from scratch
PATCH, STRIDE = 128, 64
BATCH, EPOCHS, LR, PATIENCE = 16, 50, 1e-4, 10
MODELS = ["unet", "segnet", "fcn"]
MODEL_COLOR = {"unet": "#0072B2", "segnet": "#009E73", "fcn": "#D55E00"}

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                     "savefig.dpi": 300, "savefig.bbox": "tight",
                     "pdf.fonttype": 42})

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, models, backend as K
    from sklearn.model_selection import train_test_split
except ImportError:
    raise SystemExit("TensorFlow is required for step 02. Activate the "
                     "project .venv and re-run.")


# ---------------------------------------------------------------------------
# LOSS AND METRICS
# ---------------------------------------------------------------------------
def dice_coef(y_true, y_pred):
    y_bin = tf.cast(y_pred > 0.5, tf.float32)
    inter = K.sum(y_true * y_bin)
    return 2 * inter / (K.sum(y_true) + K.sum(y_bin) + K.epsilon())


def combined_loss(y_true, y_pred):
    bce = keras.losses.binary_crossentropy(y_true, y_pred)
    return 0.5 * bce + 0.5 * (1 - dice_coef(y_true, y_pred))


def iou_metric(y_true, y_pred):
    y_bin = tf.cast(y_pred > 0.5, tf.float32)
    inter = K.sum(y_true * y_bin)
    union = K.sum(y_true) + K.sum(y_bin) - inter
    return inter / (union + K.epsilon())


# ---------------------------------------------------------------------------
# ARCHITECTURES (identical to the original study)
# ---------------------------------------------------------------------------
def conv_block(x, f):
    x = layers.Conv2D(f, 3, activation="relu", padding="same")(x)
    return layers.Conv2D(f, 3, activation="relu", padding="same")(x)


def build_unet(shape=(PATCH, PATCH, 1)):
    """Ronneberger et al. (2015): encoder-decoder with skip connections."""
    inp = layers.Input(shape)
    c1 = conv_block(inp, 64);  p1 = layers.MaxPooling2D(2)(c1)
    c2 = conv_block(p1, 128);  p2 = layers.MaxPooling2D(2)(c2)
    c3 = conv_block(p2, 256);  p3 = layers.MaxPooling2D(2)(c3)
    c4 = conv_block(p3, 512);  p4 = layers.MaxPooling2D(2)(c4)
    c5 = conv_block(p4, 1024)
    u = c5
    for c, f in zip([c4, c3, c2, c1], [512, 256, 128, 64]):
        u = layers.Conv2DTranspose(f, 2, strides=2, padding="same")(u)
        u = layers.concatenate([u, c])
        u = conv_block(u, f)
    out = layers.Conv2D(1, 1, activation="sigmoid")(u)
    return models.Model(inp, out, name="UNet")


def build_segnet(shape=(PATCH, PATCH, 1)):
    """Badrinarayanan et al. (2017), simplified: symmetric encoder-decoder
    without skip connections (up-sampling instead of pooling indices)."""
    inp = layers.Input(shape)
    x = inp
    for f in [64, 128, 256]:
        x = conv_block(x, f); x = layers.MaxPooling2D(2)(x)
    for f in [256, 128, 64]:
        x = layers.UpSampling2D(2)(x); x = conv_block(x, f)
    out = layers.Conv2D(1, 1, activation="sigmoid")(x)
    return models.Model(inp, out, name="SegNet")


def build_fcn(shape=(PATCH, PATCH, 1)):
    """Long et al. (2015), simplified: encoder + progressive up-sampling."""
    inp = layers.Input(shape)
    x = conv_block(inp, 64); x = layers.MaxPooling2D(2)(x)
    x = conv_block(x, 128); x = layers.MaxPooling2D(2)(x)
    x = conv_block(x, 256); x = layers.MaxPooling2D(2)(x)
    x = layers.Conv2D(512, 3, activation="relu", padding="same")(x)
    x = layers.UpSampling2D(8, interpolation="bilinear")(x)
    x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)
    out = layers.Conv2D(1, 1, activation="sigmoid")(x)
    return models.Model(inp, out, name="FCN")


BUILDERS = {"unet": build_unet, "segnet": build_segnet, "fcn": build_fcn}


# ---------------------------------------------------------------------------
# DATA PREPARATION
# ---------------------------------------------------------------------------
def label_path(year, season):
    path = LABEL_DIR / f"mask_{year}_{season}.tif"
    if not path.exists():
        raise SystemExit("Missing classical mask labels. Run "
                         "scripts/01_classical_extraction.py first.")
    return path


def build_patches():
    """Slide a 128-px window (stride 64) over every (MNDWI, label) pair."""
    X, Y = [], []
    for path in sorted(glob.glob(str(DATA / "MNDWI_*.tif"))):
        year, season, _ = core.parse_name(path)
        lp = label_path(year, season)
        if lp is None:
            continue
        mndwi, _, _ = core.read_mndwi(path)
        with rasterio.open(lp) as src:
            lab = (src.read(1) > 0).astype(np.float32)
        H, W = mndwi.shape
        for i in range(0, H - PATCH + 1, STRIDE):
            for j in range(0, W - PATCH + 1, STRIDE):
                xm = mndwi[i:i + PATCH, j:j + PATCH]
                if np.isnan(xm).any():
                    continue
                X.append(xm); Y.append(lab[i:i + PATCH, j:j + PATCH])
    X = np.array(X)[..., None].astype(np.float32)
    Y = np.array(Y)[..., None].astype(np.float32)
    # stratify by water fraction so the splits share the class balance
    strata = pd.qcut([y.mean() for y in Y], q=4, labels=False, duplicates="drop")
    Xtr, Xtmp, Ytr, Ytmp, _, stmp = train_test_split(
        X, Y, strata, test_size=0.30, random_state=42, stratify=strata)
    Xva, Xte, Yva, Yte = train_test_split(
        Xtmp, Ytmp, test_size=0.50, random_state=42, stratify=stmp)[:4]
    print(f"  patches: {len(Xtr)} train / {len(Xva)} val / {len(Xte)} test")
    return (Xtr, Ytr), (Xva, Yva), (Xte, Yte)


# ---------------------------------------------------------------------------
# TRAINING / LOADING
# ---------------------------------------------------------------------------
def get_model(name, train_data=None, val_data=None):
    ck = CKPT_DIR / f"{name}_best.keras"
    if not RETRAIN:
        for cand in (ck, CKPT_FALLBACK / f"{name}_best.keras"):
            if cand.exists():
                print(f"  [{name}] loading checkpoint {cand.name}")
                return keras.models.load_model(cand, compile=False), None
    print(f"  [{name}] training...")
    model = BUILDERS[name]()
    model.compile(optimizer=keras.optimizers.Adam(LR), loss=combined_loss,
                  metrics=[iou_metric, dice_coef])
    cbs = [keras.callbacks.ModelCheckpoint(ck, save_best_only=True,
                                           monitor="val_loss"),
           keras.callbacks.EarlyStopping(patience=PATIENCE,
                                         restore_best_weights=True),
           keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)]
    hist = model.fit(*train_data, validation_data=val_data, epochs=EPOCHS,
                     batch_size=BATCH, callbacks=cbs, verbose=2)
    (OUT / "dl" / f"history_{name}.json").write_text(
        json.dumps({k: [float(v) for v in vals]
                    for k, vals in hist.history.items()}))
    return model, hist.history


# ---------------------------------------------------------------------------
# EVALUATION AND INFERENCE
# ---------------------------------------------------------------------------
def evaluate(model, Xte, Yte):
    prob = model.predict(Xte, batch_size=BATCH, verbose=0)
    yp = (prob > 0.5).astype(np.uint8).ravel()
    yt = (Yte > 0.5).astype(np.uint8).ravel()
    tp = int(((yp == 1) & (yt == 1)).sum()); tn = int(((yp == 0) & (yt == 0)).sum())
    fp = int(((yp == 1) & (yt == 0)).sum()); fn = int(((yp == 0) & (yt == 1)).sum())
    return {"iou": tp / (tp + fp + fn),
            "f1": 2 * tp / (2 * tp + fp + fn),
            "accuracy": (tp + tn) / len(yt),
            "precision": tp / (tp + fp),
            "recall": tp / (tp + fn)}


def predict_full(model, arr):
    """Sliding-window inference over the full raster, averaged overlaps."""
    H, W = arr.shape
    ph = max(H, PATCH); pw = max(W, PATCH)
    ph += (-ph) % STRIDE; pw += (-pw) % STRIDE
    canvas = np.zeros((ph, pw), np.float32); canvas[:H, :W] = arr
    prob = np.zeros_like(canvas); cnt = np.zeros_like(canvas)
    patches, locs = [], []
    for i in range(0, ph - PATCH + 1, STRIDE):
        for j in range(0, pw - PATCH + 1, STRIDE):
            patches.append(canvas[i:i + PATCH, j:j + PATCH]); locs.append((i, j))
    pred = model.predict(np.array(patches)[..., None], batch_size=BATCH,
                         verbose=0)[..., 0]
    for (i, j), pr in zip(locs, pred):
        prob[i:i + PATCH, j:j + PATCH] += pr
        cnt[i:i + PATCH, j:j + PATCH] += 1
    return (prob / np.maximum(cnt, 1))[:H, :W]


def training_curves_figure(histories):
    hs = {m: h for m, h in histories.items() if h}
    if not hs:                                    # checkpoints were reused
        for m in MODELS:
            hp = OUT / "dl" / f"history_{m}.json"
            if hp.exists():
                hs[m] = json.loads(hp.read_text())
    if not hs:
        print("  [SKIP] Fig08 (no training history available - checkpoints "
              "were loaded; set RETRAIN = True to regenerate)")
        return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.6),
                                   constrained_layout=True)
    for m, h in hs.items():
        ax1.plot(h["loss"], color=MODEL_COLOR[m], lw=1.2, label=f"{m} train")
        ax1.plot(h["val_loss"], color=MODEL_COLOR[m], lw=1.2, ls="--",
                 label=f"{m} val")
        key = "iou_metric" if "iou_metric" in h else "val_iou_metric"
        ax2.plot(h.get("val_iou_metric", h.get(key, [])),
                 color=MODEL_COLOR[m], lw=1.2, label=m)
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("BCE + Dice loss")
    ax1.legend(fontsize=6); ax1.grid(alpha=0.25, lw=0.4)
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Validation IoU")
    ax2.legend(fontsize=7); ax2.grid(alpha=0.25, lw=0.4)
    for ax, letter in zip((ax1, ax2), "ab"):
        ax.text(-0.10, 1.06, f"({letter})", transform=ax.transAxes,
                fontsize=11, fontweight="bold", va="top")
    fig.savefig(FIG / "Fig08_training_curves.png")
    fig.savefig(FIG / "Fig08_training_curves.pdf")
    plt.close(fig)
    print("  [OK] Fig08_training_curves")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    print("STEP 02 - deep-learning extraction (U-Net / SegNet / FCN)")
    need_training = RETRAIN or not all(
        (CKPT_DIR / f"{m}_best.keras").exists() or
        (CKPT_FALLBACK / f"{m}_best.keras").exists() for m in MODELS)
    train = val = test = None
    if need_training:
        train, val, test = build_patches()

    histories, metrics_rows = {}, []
    for name in MODELS:
        model, hist = get_model(name, train, val)
        histories[name] = hist

        if test is None:
            _, _, test = build_patches()
        met = evaluate(model, *test)
        metrics_rows.append({"model": name, **{k: round(v, 5)
                                               for k, v in met.items()}})
        print(f"  [{name}] IoU = {met['iou']:.4f}  F1 = {met['f1']:.4f}")

        # full-raster inference + identical vectorisation as classical
        records = []
        for path in sorted(glob.glob(str(DATA / "MNDWI_*.tif"))):
            year, season, date = core.parse_name(path)
            mndwi, transform, crs = core.read_mndwi(path)
            mask = (predict_full(model, mndwi) > 0.5).astype(np.uint8)
            line, _ = core.extract_shoreline(mask, transform, crs,
                                             valid=np.isfinite(mndwi))
            if line is None:
                print(f"    [WARN] {name}: no shoreline {year} {season}")
                continue
            records.append({"geometry": line, "year": year, "season": season,
                            "date": date, "length_m": round(line.length, 1)})
        gdf = core.save_shorelines(records, SHORE_DIR, name)
        print(f"  [{name}] {len(gdf)} shorelines -> {SHORE_DIR}")

    pd.DataFrame(metrics_rows).to_csv(TAB / "pixel_metrics.csv", index=False)
    json.dump({r["model"]: r for r in metrics_rows},
              open(OUT / "dl" / "test_metrics.json", "w"), indent=2)
    training_curves_figure(histories)
    print("\nStep 02 finished.")


if __name__ == "__main__":
    main()
