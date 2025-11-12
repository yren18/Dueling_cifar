import os
from PIL import Image
import matplotlib.pyplot as plt

results_dir = "results"
prefix = "1_"  
out_path = os.path.join(results_dir, f"{prefix}figure.png")

order = [
    f"{prefix}Original_['cat'].png",   # (a)
    f"{prefix}PGD_['ship'].png",       # (b)
    f"{prefix}RDueling_['ship'].png",  # (c)
    f"{prefix}RGD_['ship'].png",       # (d)
    f"{prefix}RZO_['ship'].png",       # (e)
    f"{prefix}loss_vs_iteration.png"   # (f)
]

labels = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"]
titles = ["Original", "PGD", "R-Dueling", "RGD", "RZO", "Loss vs Iteration"]

imgs = []
for fn in order:
    path = os.path.join(results_dir, fn)
    if os.path.exists(path):
        imgs.append(Image.open(path).convert("RGB"))
    else:
        imgs.append(None)
        print(f"[Warning] Missing: {fn}")


fig, axes = plt.subplots(1, 6, figsize=(18, 3))
for i, ax in enumerate(axes):
    img = imgs[i]
    if img is not None:
        ax.imshow(img)
        ax.set_title(titles[i], fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.text(0.5, -0.15, labels[i], transform=ax.transAxes, ha="center", fontsize=10)

plt.tight_layout()
plt.savefig(out_path, dpi=200, bbox_inches="tight")
plt.close()
print(f"Saved combined figure: {out_path}")
