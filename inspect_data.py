"""Step 1: data inspection. Prints split sizes/class counts, checks for duplicates and
bad values, builds the IID client partition, saves tables + plots to results/."""
import argparse, json, os, numpy as np, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from common import *

ap = argparse.ArgumentParser()
ap.add_argument("--clients", type=int, default=20); ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--dataset", default="organamnist",
                choices=["organamnist", "organcmnist", "organsmnist", "breastmnist", "tissuemnist"])
ap.add_argument("--synthetic", action="store_true"); a = ap.parse_args()
os.makedirs("results", exist_ok=True)

# NUM_CLASSES/LABELS from common.py are organ-specific (11 classes); for any other dataset
# (breast, tissue) look up the real class count/names from medmnist's own metadata instead.
if a.synthetic:
    num_classes, labels = NUM_CLASSES, LABELS   # common.py's synthetic generator is organ-shaped only
else:
    import medmnist
    info = medmnist.INFO[a.dataset]
    num_classes = len(info["label"])
    labels = [info["label"][str(i)] for i in range(num_classes)]

data = {s: load_split(s, a.synthetic, name=a.dataset) for s in ["train", "val", "test"]}
counts = {}
for s, (x, y) in data.items():
    counts[s] = np.bincount(y.numpy(), minlength=num_classes).tolist()
    print(f"{s:5s} n={len(x):6d}  shape={tuple(x.shape[1:])}  min={x.min():.2f} max={x.max():.2f}  "
          f"NaN={bool(torch.isnan(x).any())}")
    print("      class counts:", counts[s])
xtr, ytr = data["train"]
# cleaning checks: exact duplicate images inside train, and train/test overlap
h = lambda x: {hash(t.numpy().tobytes()) for t in x}
htr, hte = h(xtr), h(data["test"][0])
print(f"duplicate train images: {len(xtr) - len(htr)} | train/test exact overlap: {len(htr & hte)}")

parts = iid_partition(ytr, a.clients, a.seed)
table = np.array([np.bincount(ytr.numpy()[p], minlength=num_classes) for p in parts])
print(f"\nIID partition, {a.clients} clients, seed {a.seed}: images per client "
      f"min={table.sum(1).min()} max={table.sum(1).max()}")
np.savetxt(f"results/{a.dataset}_client_class_counts.csv", np.c_[np.arange(1, a.clients + 1), table.sum(1), table],
           fmt="%d", delimiter=",", header="client,images," + ",".join(labels))
json.dump(counts, open(f"results/{a.dataset}_split_class_counts.json", "w"))

fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
# distinct line style/marker/width per split so lines stay visible even when splits are
# near-perfectly stratified and would otherwise sit exactly on top of each other
STYLE = {"train": dict(color="#1f77b4", linestyle="-", marker="o", markersize=7, linewidth=2.5, zorder=3),
         "val":   dict(color="#ff7f0e", linestyle="--", marker="s", markersize=5.5, linewidth=1.8, zorder=2),
         "test":  dict(color="#2ca02c", linestyle=":", marker="^", markersize=4, linewidth=1.4, zorder=1)}
for s in ["train", "val", "test"]:
    ax[0].plot(np.array(counts[s]) / sum(counts[s]), label=s, alpha=0.9, **STYLE[s])
ax[0].set_xticks(range(num_classes)); ax[0].set_xlabel("class id"); ax[0].set_ylabel("fraction of split")
ax[0].set_title("Class balance per official split"); ax[0].legend()
bottom = np.zeros(a.clients)
for c in range(num_classes):
    ax[1].bar(range(1, a.clients + 1), table[:, c], bottom=bottom, label=str(c)); bottom += table[:, c]
ax[1].set_xlabel("client"); ax[1].set_ylabel("training images"); ax[1].set_title("Client partition (IID)")
ax[1].legend(ncol=2, fontsize=6, title="class")
plt.tight_layout(); plt.savefig(f"results/{a.dataset}_data_overview.png", dpi=150)
print("saved results/client_class_counts.csv, split_class_counts.json, data_overview.png")
