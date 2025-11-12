import os
from PIL import Image
import matplotlib.pyplot as plt

# --- Configuration ---
results_dir = "results"
prefix = "1_"  # change this prefix to select another example
out_path = os.path.join(results_dir, f"{prefix}figure.png")

# Order of images (a)-(f)
order = [
    f"{prefix}Original_['cat'].png",     # (a)
    f"{prefix}PGD_['ship'].png",         # (b)
    f"{prefix}RGD_['ship'].png",         # (c)
    f"{prefix}RZO_['ship'].png",         # (d)
    f"{prefix}RDueling_['ship'].png",    # (e)
    f"{prefix}loss_vs_iteration.png"     # (f)
]

labels = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"]
titles = [
    "Original",
    "PGD (First-order)",
    "RGD (First-order)",
    "RZO (Zeroth-order)",
    "R-Dueling (Comparison)",
    "Loss vs Iteration"
]

# --- Load images ---
imgs = []
for fn in order:
    path = os.path.join(results_dir, fn)
    if os.path.exists(path):
        imgs.append(Image.open(path).convert("RGB"))
    else:
        imgs.append(None)
        print(f"[Warning] Missing: {fn}")

# --- Create figure ---
fig, axes = plt.subplots(1, 6, figsize=(18, 3))

for i, ax in enumerate(axes):
    img = imgs[i]
    if img is not None:
        ax.imshow(img)
        ax.set_title(titles[i], fontsize=9)
    else:
        ax.text(0.5, 0.5, "Missing", ha="center", va="center", fontsize=9, color="red")
        ax.set_facecolor("lightgray")

    ax.set_xticks([])
    ax.set_yticks([])
    ax.text(0.5, -0.15, labels[i], transform=ax.transAxes,
            ha="center", fontsize=10, fontweight="bold")

plt.tight_layout()
plt.savefig(out_path, dpi=200, bbox_inches="tight")
plt.close()
print(f"[Saved] Combined figure → {out_path}")
