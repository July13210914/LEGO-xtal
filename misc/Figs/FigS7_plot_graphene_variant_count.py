# topology.py — fast labeling from DB 'topology' field only
from collections import Counter
from ase.db import connect
import csv
import os
import matplotlib.pyplot as plt

DB_PATH = "../../data/source/sp2_sacada.db"
CSV_OUT = "lego_sp2_graphene_stats.csv"  
OUT_PNG = "energy_vs_density_mace.png"
OUT_POINTS_CSV = "energy_vs_density_points_mace.csv"
# Teal palette to match the paper's figures
COLOR = {
    "graphene_stacking_variant": "#0B4F6C",  # deep blue-teal
    "defective_graphene_like":   "#33FFDA",  # bright aqua-teal
    "non_graphene_topology":     "#B7D5D4",  # light teal-gray
}

# Map DB 'topology' string to our classes
# - all tokens 'hcb' -> graphene_stacking_variant
# - 'hcb' mixed with non-hex tokens -> defective_graphene_like
# - otherwise -> non_graphene_topology
NON_HEX_TOKENS = {"hnd", "hnc", "hne", "hna", "cae", "car", "dhh"}

def label_from_topology_string(topo_str):
    if not topo_str:
        return "non_graphene_topology"
    s = str(topo_str).strip().lower()
    toks = [tok.split("(")[0].strip() for tok in s.split("-") if tok.strip()]
    if not toks:
        return "non_graphene_topology"
    if all(tok == "hcb" for tok in toks):
        return "graphene_stacking_variant"
    if "hcb" in toks and any(tok in NON_HEX_TOKENS for tok in toks):
        return "defective_graphene_like"
    return "non_graphene_topology"

def main():
    rows = []
    points = []  # (density, energy, label, id)
    with connect(DB_PATH) as db:
        for row in db.select():
            topo = row.get("topology", None)
            label = label_from_topology_string(topo)
            atoms = row.toatoms()  # just to get formula/natoms
            rows.append({
                "id": row.id,
                "formula": atoms.get_chemical_formula(),
                "natoms": len(atoms),
                "space_group_number": row.get("space_group_number", None),
                "topology": topo,
                "label": label,
            })
            dens = row.get("density", None)
            e = row.get("mace_energy", None)
            if dens is not None and e is not None:
                points.append((dens, e, label, row.id))

    # counts
    counts = Counter(r["label"] for r in rows)
    total = len(rows)
    def pct(n): return 0.0 if total == 0 else 100.0 * n / total

    print("=== Graphene-like classification (topology-only) ===")
    for k in ["graphene_stacking_variant", "defective_graphene_like", "non_graphene_topology"]:
        print(f"{k:30s}: {counts.get(k,0):6d} ({pct(counts.get(k,0)):5.1f}%)")
    print(f"Total structures: {total}")

    # write CSV (compact)
    fieldnames = ["id", "formula", "natoms", "space_group_number", "topology", "label"]
    with open(CSV_OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote labels to {CSV_OUT}")

    # --- Plot: Energy (mace) vs Density, colored by topology-derived label ---
    if points:
        from collections import Counter as _Ctr
        _classes = _Ctr(p[2] for p in points)
        print("Plotted points by class:", dict(_classes))
        plt.figure(figsize=(10, 7))
        
        # Define different markers for each category
        markers = {
            "graphene_stacking_variant": "D",  # Diamond
            "defective_graphene_like": "^",    # Triangle
            "non_graphene_topology": "o"       # Circle
        }
        
        # Define formal labels for research paper
        formal_labels = {
            "graphene_stacking_variant": "Graphene Stacking Variants",
            "defective_graphene_like": "Defective Graphene-like",
            "non_graphene_topology": "Non-graphene Topology"
        }
        
        for k in ["graphene_stacking_variant", "defective_graphene_like", "non_graphene_topology"]:
            xs = [p[0] for p in points if p[2] == k]
            ys = [p[1] for p in points if p[2] == k]
            if xs:
                plt.scatter(xs, ys, s=20, alpha=0.8, 
                           label=f"{formal_labels[k]} (n={len(xs)})",
                           c=COLOR.get(k, "#888888"), edgecolors="#274043", linewidths=0.3,
                           marker=markers.get(k, "o"))
        plt.ylim(-9.4, None)
        plt.xlim(None, 3.1)
        plt.xlabel("Density (g/cm³)")
        plt.ylabel("MACE energy (eV/atom)")
        plt.title("Energy vs Density Plot of Structures in LEGO sp2 Database", fontsize=12)
        plt.legend(frameon=True, fontsize=10, fancybox=True, shadow=True, 
                  framealpha=0.9, edgecolor='black', facecolor='white')
        plt.tight_layout()
        plt.savefig(OUT_PNG, dpi=300)
        print(f"Saved plot → {OUT_PNG}")
        # Save plotted points for SI/reproducibility
        import csv as _csv
        with open(OUT_POINTS_CSV, "w", newline="") as f:
            w = _csv.writer(f)
            w.writerow(["id", "label", "density", "mace_energy"])
            for dens, e, lab, rid in points:
                w.writerow([rid, lab, dens, e])
        print(f"Saved plotted data → {OUT_POINTS_CSV}")
    else:
        print("No points had both density and mace_energy; skipping plot.")

if __name__ == "__main__":
    main()
