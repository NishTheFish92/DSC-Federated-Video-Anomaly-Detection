"""Every multi-seed experiment in the paper (Table 2, Figure 2 and Table 3).

For each seed in SEEDS, all with the same 30-epoch budget, this trains FedAvg (AUC recorded
after every round), centralised and single-site models. For each seed in EXTRA_SEEDS it also trains:
  - FedProx with each mu in PROX_MUS
  - FedAvg with each site's clips split into several clients (K = 4 and K = 8)
  - FedAvg over the two halves of one site's clips (two clients that film the same scene)
  - FedAvg and centralised models that take stacks of STACK consecutive frames
Results go to results/experiments.json after each seed (3 to 17 min per seed on a GTX 1650 Ti).
    uv run run_experiments.py            # run everything, then print the summary
    uv run run_experiments.py --summary  # only print the summary of the saved results
"""
import json
import sys
import time

from src.centralized import train_centralized
from src.config import (CLIENTS_PER_SITE, DEVICE, EXTRA_SEEDS, LOCAL_EPOCHS, PROX_MUS, RESULTS_DIR,
                        ROUNDS, SEEDS, SITES, STACK, set_seed)
from src.data import client_loaders, test_clips, train_loader
from src.evaluate import evaluate_sites, site_extra_metrics
from src.federated import run_fedavg
from src.model import AutoEncoder
from src.report import summarize
from src.train_local import train

OUT = RESULTS_DIR / "experiments.json"


def timed(fn):
    """Run fn() and return (its result, seconds taken)."""
    start = time.time()
    result = fn()
    return result, time.time() - start


def fedavg(seed, loaders, tests, channels=1, mu=0.0, rounds=None):
    """One federated run from a fresh seed; returns its AUCs (and per-round AUCs if asked)."""
    set_seed(seed)
    after = (lambda m: rounds.append(evaluate_sites(m, tests))) if rounds is not None else None
    model, _ = run_fedavg(loaders, channels, mu, after)
    return model, evaluate_sites(model, tests)


def single_site(seed, loader, tests):
    """One model trained on one site's frames only, for the full 30 epochs."""
    set_seed(seed)
    model = AutoEncoder().to(DEVICE)
    train(model, loader, ROUNDS * LOCAL_EPOCHS)
    return evaluate_sites(model, tests)


def table2_runs(seed, loaders, tests):
    """FedAvg (with AUC after every round), centralised and single-site models."""
    r = {"seed": seed, "fedavg_rounds": []}
    (fed, r["fedavg"]), r["fedavg_time"] = timed(
        lambda: fedavg(seed, loaders, tests, rounds=r["fedavg_rounds"]))
    set_seed(seed)
    cen, r["centralised_time"] = timed(lambda: train_centralized(loaders))
    r["centralised"] = evaluate_sites(cen, tests)
    for name, model in (("fedavg", fed), ("centralised", cen)):
        r[name + "_extra"] = {s: site_extra_metrics(model, c) for s, c in tests.items()}
    for site in SITES:
        r["single_" + site] = single_site(seed, loaders[site], tests)
    return r


def extra_runs(seed, loaders, stack_loaders, tests):
    """FedProx, more clients, same-scene clients and frame stacks."""
    r = {}
    for mu in PROX_MUS:
        r[f"fedprox_{mu}"] = fedavg(seed, loaders, tests, mu=mu)[1]
    for n in CLIENTS_PER_SITE:
        clients = {f"{s}_{i}": l for s, p in SITES.items() for i, l in enumerate(client_loaders(p, n))}
        r[f"fedavg_k{len(clients)}"] = fedavg(seed, clients, tests)[1]
    for site, path in SITES.items():
        halves = dict(enumerate(client_loaders(path, 2)))
        r["same_scene_" + site] = fedavg(seed, halves, tests)[1]

    r["stack_fedavg"] = fedavg(seed, stack_loaders, tests, channels=STACK)[1]
    set_seed(seed)
    r["stack_centralised"] = evaluate_sites(train_centralized(stack_loaders, STACK), tests)
    return r


def main():
    if "--summary" not in sys.argv:
        loaders = {site: train_loader(path) for site, path in SITES.items()}
        stack_loaders = {site: train_loader(path, STACK) for site, path in SITES.items()}
        tests = {site: test_clips(path) for site, path in SITES.items()}
        results = []
        for seed in SEEDS:
            print(f"=== seed {seed}")
            r = table2_runs(seed, loaders, tests)
            if seed in EXTRA_SEEDS:
                r.update(extra_runs(seed, loaders, stack_loaders, tests))
            results.append(r)
            RESULTS_DIR.mkdir(exist_ok=True)
            OUT.write_text(json.dumps(results, indent=1))
    summarize(json.loads(OUT.read_text()))


if __name__ == "__main__":
    main()
