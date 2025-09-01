# topology.py — fast labeling from DB 'topology' field only
from collections import Counter
from ase.db import connect
import csv
import os
import matplotlib.pyplot as plt

DB_PATH = "../../data/source/lego-sp2.db"
CSV_OUT = "lego_sp2_graphene_stats.csv"
OUT_PNG = "FigS5.pdf"
OUT_POINTS_CSV = "energy_vs_density_points_mace.csv"
# Teal palette to match the paper's figures
COLORS = [
    "#0B4F6C",  # deep blue-teal
    "#33FFDA",  # bright aqua-teal
    "#B7D5D4",  # light teal-gray
    "gray",
]

# Map DB 'topology' string to our classes
# - all tokens 'hcb' -> graphene_stacking_variant
# - 'hcb' mixed with non-hex tokens -> defective_graphene_like
# - otherwise -> non_graphene_topology
NON_HEX_TOKENS = {"hnd", "hnc", "hne", "hna", "cae", "car", "dhh"}

def label_from_topology_string(topo_str, dim):
    if dim == 2:
        s = str(topo_str).strip().lower()
        toks = [tok.split("(")[0].strip() for tok in s.split("-") if tok.strip()]
        if all(tok == "hcb" for tok in toks):
            return 0  # hcb variant
        elif "hcb" in toks: #and any(tok in NON_HEX_TOKENS for tok in toks):
            return 1  # hcb + ***
        else:
            return 2  # non-hcb
    else:
        return 3

def main():
    rows = []
    points = []  # (density, energy, label, id)
    with connect(DB_PATH) as db:
        for row in db.select():
            topo = row.get("topology", None)
            dim = row.get("dimension", None)
            label = label_from_topology_string(topo, dim)
            dens = row.get("density", None)
            e = row.get("mace_energy", None)
            if label < 3:
                print(topo, dim, label, dens, e)
            if dens is not None and e is not None:
                points.append((dens, e, label, row.id))

    # counts
    counts = Counter(r["label"] for r in rows)
    total = len(rows)

    from collections import Counter as _Ctr
    _classes = _Ctr(p[2] for p in points)
    print("Plotted points by class:", dict(_classes))
    plt.figure(figsize=(10, 5))

    # Define different markers for each category
    markers = ["*", "d", "s", "o"]

    # Define formal labels for research paper
    formal_labels = ["HCB", "HCB+other", "Other 2D", "non-2D"]

    for k in [0, 1, 2, 3]:
        xs = [p[0] for p in points if p[2] == k]
        ys = [p[1] for p in points if p[2] == k]
        if xs:
            s = 35 if k > 0 else 80
            alpha = 0.8 if k<3 else 0.4
            plt.scatter(xs, ys, s=s, alpha=alpha,
                       label=f"{formal_labels[k]} ({len(xs)})",
                       c=COLORS[k], edgecolors="#274043", #linewidths=0.3,
                       marker=markers[k])
    plt.ylim(-9.4, None)
    plt.xlim(None, 3.0)
    plt.xlabel("Density (g/cm³)", fontsize=15)
    plt.ylabel("MACE Energy (eV/atom)", fontsize=15)
    #plt.title("Energy vs Density Plot of Structures in LEGO sp2 Database", fontsize=15)
    plt.legend(frameon=True, fontsize=15, fancybox=True, shadow=True,
               loc=3,
              framealpha=0.9, edgecolor='black', facecolor='white')
    plt.tight_layout()
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.savefig(OUT_PNG)

if __name__ == "__main__":
    main()
