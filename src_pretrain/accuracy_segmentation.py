"""
Cadre d'évaluation — un graphe par structure.

Les résultats sont lus depuis des CSV séparés par des espaces, de la forme :

    time mDice cortex background CSF WhiteMatter Cortex Ventricles
    92   0.9098 0.9007 0.9988 0.9603 0.9779 0.9007 0.7114
    99.0 0.9209 0.8989 ...
    ...

Pour CHAQUE structure (colonne autre que `time`), on produit deux figures :
  - Dice absolu par méthode        -> figures/dice_<structure>.png
  - Gain vs référence (points %)   -> figures/gain_<structure>.png

La colonne `time` fournit les labels de l'axe des x.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- Données (à remplacer) : un CSV par méthode ---
paths = {
    "Pretrain": "/media/florian/Nouveau nom/Result_joint/pretrain/results/dice_seg.csv",  # référence
    "Pure seg": "/media/florian/Nouveau nom/Result_joint/Simple/seg/results/dice_seg.csv",  # référence
    "Seg/Int": "/media/florian/Nouveau nom/Result_joint/All/seg_int/results/dice_seg.csv",  # référence
    "Replay": "/media/florian/Nouveau nom/Result_joint/replay/results/dice_seg.csv",  # référence
    "Fonctionnel": "/media/florian/Nouveau nom/Result_joint/distillation/results/dice_seg.csv",  # référence
}

REFERENCE = "Pretrain"  # méthode servant de référence pour le gain
OUTDIR = "figures"

couleurs = {"Seg/Int": "#2a78d6", "Pure seg": "#1baf7a", "Weight": "#eda100",
            "Last anchor": "#d62a2a", "Pretrain": "#888780", "Replay": "#00ffdd",
            "Fonctionnel": "#ff0e87"}

# --- Lecture des CSV ---
data = {m: pd.read_csv(p, sep=r"\s+") for m, p in paths.items()}

ref_df = data[REFERENCE]
time_points = [f"{t:g}" for t in ref_df["time"].to_numpy()]
x = np.arange(len(time_points))

# Structures = toutes les colonnes sauf `time`, dédupliquées (cortex/Cortex identiques)
structures, seen = [], set()
for c in ref_df.columns:
    if c == "time":
        continue
    if c.lower() in seen:
        continue
    seen.add(c.lower())
    structures.append(c)


def grouped_bar(values_by_method, ylabel, title, outfile, ylim=None, zero_line=False):
    """Barres groupées par time point, une couleur par méthode."""
    methods = list(values_by_method.keys())
    width = 0.7 / len(methods)
    fig, ax = plt.subplots(figsize=(8.5, 4.6))

    # bandes de fond alternées + séparateurs
    for i in range(len(time_points)):
        if i % 2 == 1:
            ax.axvspan(i - 0.5, i + 0.5, color="#f1efe8", zorder=0)
    for b in x[:-1] + 0.5:
        ax.axvline(b, color="#d3d1c7", lw=0.8, zorder=1)

    for i, m in enumerate(methods):
        offset = (i - (len(methods) - 1) / 2) * width
        ax.bar(x + offset, values_by_method[m], width,
               color=couleurs.get(m, None), edgecolor="white",
               linewidth=0.6, label=m, zorder=3)

    if zero_line:
        ax.axhline(0, ls="--", lw=1.3, color="#888780", zorder=2)
        ax.text(len(time_points) - 0.5, 0.04, f"{REFERENCE} (référence)",
                ha="right", va="bottom", fontsize=9, color="#888780")

    ax.set_xticks(x);
    ax.set_xticklabels(time_points)
    ax.set_xlim(-0.5, len(time_points) - 0.5)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.grid(axis="y", color="#e1e0d9", lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, ncol=len(methods), loc="upper left", bbox_to_anchor=(0, 1.0))
    fig.tight_layout()
    fig.savefig(outfile, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure enregistrée : {outfile}")


os.makedirs(OUTDIR, exist_ok=True)

for struct in structures:
    ref_vals = ref_df[struct].to_numpy()

    # --- Dice absolu (toutes les méthodes) ---
    abs_vals = {m: data[m][struct].to_numpy() * 100 for m in paths}
    grouped_bar(
        abs_vals,
        ylabel="Dice (%)",
        title=f"Dice absolu — {struct}",
        outfile=os.path.join(OUTDIR, f"dice_seg_{struct}.png"),
        ylim=(0, 100),
    )

    # --- Gain vs référence (méthodes hors référence) ---
    gain_vals = {m: (data[m][struct].to_numpy() - ref_vals) * 100
                 for m in paths if m != REFERENCE}
    grouped_bar(
        gain_vals,
        ylabel=f"Gain de Dice vs {REFERENCE} (points %)",
        title=f"Gain par méthode — {struct}",
        outfile=os.path.join(OUTDIR, f"gain_seg_{struct}.png"),
        zero_line=True,
    )

# --- Données (à remplacer) : un CSV par méthode ---
paths = {
    "Pretrain": "/media/florian/Nouveau nom/Result_joint/pretrain/results/dice_reg.csv",  # référence
    "Pure seg": "/media/florian/Nouveau nom/Result_joint/Simple/seg/results/dice_reg.csv",  # référence
    "Seg/Int": "/media/florian/Nouveau nom/Result_joint/All/seg_int/results/dice_reg.csv",  # référence
    "Replay": "/media/florian/Nouveau nom/Result_joint/replay/results/dice_reg.csv",  # référence
    "Fonctionnel": "/media/florian/Nouveau nom/Result_joint/distillation/results/dice_reg.csv",  # référence
}


# --- Lecture des CSV ---
data = {m: pd.read_csv(p, sep=r"\s+") for m, p in paths.items()}

ref_df = data[REFERENCE]
time_points = [f"{t:g}" for t in ref_df["time"].to_numpy()]
x = np.arange(len(time_points))

# Structures = toutes les colonnes sauf `time`, dédupliquées (cortex/Cortex identiques)
structures, seen = [], set()
for c in ref_df.columns:
    if c == "time":
        continue
    if c.lower() in seen:
        continue
    seen.add(c.lower())
    structures.append(c)


os.makedirs(OUTDIR, exist_ok=True)

for struct in structures:
    ref_vals = ref_df[struct].to_numpy()

    # --- Dice absolu (toutes les méthodes) ---
    abs_vals = {m: data[m][struct].to_numpy() * 100 for m in paths}
    grouped_bar(
        abs_vals,
        ylabel="Dice (%)",
        title=f"Dice absolu — {struct}",
        outfile=os.path.join(OUTDIR, f"dice_reg_{struct}.png"),
        ylim=(0, 100),
    )

    # --- Gain vs référence (méthodes hors référence) ---
    gain_vals = {m: (data[m][struct].to_numpy() - ref_vals) * 100
                 for m in paths if m != REFERENCE}
    grouped_bar(
        gain_vals,
        ylabel=f"Gain de Dice vs {REFERENCE} (points %)",
        title=f"Gain par méthode — {struct}",
        outfile=os.path.join(OUTDIR, f"gain_reg_{struct}.png"),
        zero_line=True,
    )


# --- Données (à remplacer) : un CSV par méthode ---
paths = {
    "Pretrain": "/media/florian/Nouveau nom/Result_joint/pretrain/results/gi_reg.csv",  # référence
    "Pure seg": "/media/florian/Nouveau nom/Result_joint/Simple/seg/results/gi_reg.csv",  # référence
    "Seg/Int": "/media/florian/Nouveau nom/Result_joint/All/seg_int/results/gi_reg.csv",  # référence
    "Replay": "/media/florian/Nouveau nom/Result_joint/replay/results/gi_reg.csv",  # référence
    "Fonctionnel": "/media/florian/Nouveau nom/Result_joint/distillation/results/gi_reg.csv",  # référence
}

# --- Lecture des CSV ---
data = {m: pd.read_csv(p, sep=r"\s+") for m, p in paths.items()}

ref_df = data[REFERENCE]
time_points = [f"{t:g}" for t in ref_df["time"].to_numpy()]
x = np.arange(len(time_points))

# Structures = toutes les colonnes sauf `time`, dédupliquées (cortex/Cortex identiques)
structures, seen = [], set()
for c in ref_df.columns:
    if c == "time":
        continue
    if c.lower() in seen:
        continue
    seen.add(c.lower())
    structures.append(c)


os.makedirs(OUTDIR, exist_ok=True)

for struct in structures:
    ref_vals = ref_df[struct].to_numpy()

    # --- Dice absolu (toutes les méthodes) ---
    abs_vals = {m: data[m][struct].to_numpy()  for m in paths}
    grouped_bar(
        abs_vals,
        ylabel="GI",
        title=f"GI",
        outfile=os.path.join(OUTDIR, f"gi_reg_{struct}.png"),
        ylim=(0, 2),
    )



# --- Données (à remplacer) : un CSV par méthode ---
paths = {
    "Pretrain": "/media/florian/Nouveau nom/Result_joint/pretrain/results/gi.csv",  # référence
    "Pure seg": "/media/florian/Nouveau nom/Result_joint/Simple/seg/results/gi.csv",  # référence
    "Seg/Int": "/media/florian/Nouveau nom/Result_joint/All/seg_int/results/gi.csv",  # référence
    "Replay": "/media/florian/Nouveau nom/Result_joint/replay/results/gi.csv",  # référence
    "Fonctionnel": "/media/florian/Nouveau nom/Result_joint/distillation/results/gi.csv",  # référence
}

# --- Lecture des CSV ---
data = {m: pd.read_csv(p, sep=r"\s+") for m, p in paths.items()}

ref_df = data[REFERENCE]
time_points = [f"{t:g}" for t in ref_df["time"].to_numpy()]
x = np.arange(len(time_points))

# Structures = toutes les colonnes sauf `time`, dédupliquées (cortex/Cortex identiques)
structures, seen = [], set()
for c in ref_df.columns:
    if c == "time":
        continue
    if c.lower() in seen:
        continue
    seen.add(c.lower())
    structures.append(c)


os.makedirs(OUTDIR, exist_ok=True)

for struct in structures:
    ref_vals = ref_df[struct].to_numpy()

    # --- Dice absolu (toutes les méthodes) ---
    abs_vals = {m: data[m][struct].to_numpy() for m in paths}
    grouped_bar(
        abs_vals,
        ylabel="GI",
        title=f"GI",
        outfile=os.path.join(OUTDIR, f"gi_seg_{struct}.png"),
        ylim=(0, 2),
    )

