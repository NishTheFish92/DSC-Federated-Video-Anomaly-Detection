"""Print the paper's tables from the saved multi-seed results (results/experiments.json)."""
import numpy as np

from src.config import CLIENTS_PER_SITE, PROX_MUS, SITES
from src.stats import paired_comparison

KEYS = list(SITES) + ["mean"]


def column(results, method, key):
    """One AUC per seed, e.g. column(results, "fedavg", "site1")."""
    return [r[method][key] for r in results]


def mean_sd(values):
    return f"{np.mean(values):.3f} ± {np.std(values, ddof=1):.3f}"


def print_comparison(label, a, b):
    c = paired_comparison(a, b)
    lo, hi = c["ci95"]
    print(f"  {label:34} diff {c['mean_diff']:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]"
          f"  t-test p={c['p_ttest']:.3f}  TOST(±0.02) p={c['p_tost']:.3f}")


def print_rows(results, methods):
    print(f"  {'method':22}" + "".join(f"{k:>16}" for k in KEYS))
    for m in methods:
        print(f"  {m:22}" + "".join(f"{mean_sd(column(results, m, k)):>16}" for k in KEYS))


def summarize(results):
    print(f"\n{len(results)} seeds: {[r['seed'] for r in results]}")
    print("\nTable 2: frame-level ROC-AUC, mean ± sd")
    print_rows(results, ["single_site1", "single_site2", "centralised", "fedavg"])

    print("\nCentralised minus federated (paired over seeds)")
    for k in KEYS:
        print_comparison(k, column(results, "fedavg", k), column(results, "centralised", k))
    print("Federated minus single-site, own site")
    for site in SITES:
        print_comparison(site, column(results, "single_" + site, site), column(results, "fedavg", site))

    rounds = np.array([[x["mean"] for x in r["fedavg_rounds"]] for r in results])
    print("\nFigure 2: mean AUC per round", rounds.mean(0).round(3), "sd", rounds.std(0, ddof=1).round(3))
    first = results[0]
    for m in ("fedavg", "centralised"):
        for site in SITES:
            x = first[m + "_extra"][site]
            print(f"  {m:12} {site} seed {first['seed']}: PR-AUC {x['pr_auc']:.3f}, EER {x['eer']:.3f}")

    results = [r for r in results if "stack_fedavg" in r]  # Table 3 uses EXTRA_SEEDS only
    print(f"\nTable 3 seeds: {[r['seed'] for r in results]}")
    variants = [f"fedprox_{mu}" for mu in PROX_MUS]
    variants += [f"fedavg_k{2 * n}" for n in CLIENTS_PER_SITE] + ["stack_fedavg", "stack_centralised"]
    print("Table 3: extra studies, mean ± sd")
    print_rows(results, ["fedavg", "centralised"] + variants)
    print("Variant minus FedAvg, mean AUC")
    for m in variants[:-2]:
        print_comparison(m, column(results, "fedavg", "mean"), column(results, m, "mean"))
    print("Centralised minus federated, mean AUC")
    for m in variants[:-2]:
        print_comparison(m, column(results, m, "mean"), column(results, "centralised", "mean"))
    print_comparison("frame stacks", column(results, "stack_fedavg", "mean"),
                     column(results, "stack_centralised", "mean"))
    print("Frame stacks minus single frames")
    for m in ("fedavg", "centralised"):
        for k in KEYS:
            print_comparison(f"{m} {k}", column(results, m, k), column(results, "stack_" + m, k))

    print("\nSame-scene clients: one site's clips split in two, tested on that site")
    for site in SITES:
        same = column(results, "same_scene_" + site, site)
        pooled = column(results, "single_" + site, site)
        print(f"  {site}: federated halves {mean_sd(same)}, pooled site {mean_sd(pooled)}")
        print_comparison(f"{site} pooled minus federated", same, pooled)
