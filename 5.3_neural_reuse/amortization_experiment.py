"""Amortization experiment for KAN reusability (Section 5.3).

Tests the economic argument for KAN weight reuse: the source pre-training
cost is paid ONCE, then amortized across every target problem it warm-starts.
For each category, a small source problem is pre-trained, and its KAN weights
are transferred to a ladder of larger same-density targets (exact 2x and 4x
grid scalings). Each target is solved two ways with the same L-BFGS budget:

    scratch   : random init, full budget            (the no-reuse baseline)
    transfer  : source KAN weights copied in, then fine-tuned

The headline comparison is total wall-clock across all targets:

    sum(scratch_i)   vs   pretrain + sum(effective_transfer_i)

where effective transfer time is the (approximate) time at which the
fine-tuned run first matched the scratch run's final compliance, or its full
budget if it never did.

Outputs (per --out directory):
    <category>/<target>_designs.png    final topologies (scratch / zero-shot / fine-tuned)
    <category>/<target>_curves.png     compliance-vs-time curves with crossover marked
    results.json                       all numbers, machine-readable
    summary.md                         human-readable tables

Designed for headless HPC runs (Agg backend, argparse, no widgets):

    python amortization_experiment.py --pretrain_steps 100 --steps 300
    python amortization_experiment.py --categories mbb_beam --steps 500
"""

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from neural_structural_optimization import problems, topo_api
import models as pt

# Source problem and same-density exact-2x / 4x targets per category.
AMORTIZATION_SETS = {
    "mbb_beam": {
        "source": "mbb_beam_96x32_0.5",
        "targets": ["mbb_beam_192x64_0.5", "mbb_beam_384x128_0.5"],
    },
    "cantilever_beam_full": {
        "source": "cantilever_beam_full_96x32_0.4",
        "targets": ["cantilever_beam_full_192x64_0.4",
                     "cantilever_beam_full_384x128_0.4"],
    },
}


def build_model(problem_name, seed, kan_layers, grid, k):
    args = topo_api.specified_task(problems.PROBLEMS_BY_NAME[problem_name])
    return pt.BaseKANModel(seed=seed, args=args, kan_layers=kan_layers,
                            grid=grid, k=k)


def train_timed(model, steps, progress_every=25):
    t0 = time.time()
    ds = pt.train_lbfgs(model, steps, progress_every=progress_every)
    elapsed = time.time() - t0
    losses = ds.loss.values
    best_step = int(np.nanargmin(losses))
    compliance = float(np.nanmin(losses))
    design = np.clip(ds.design.isel(step=best_step).values, 0.0, 1.0)
    return compliance, design, losses, elapsed


def zero_shot_eval(model):
    with torch.no_grad():
        logits = model()
        compliance = float(model.loss(logits).item())
        design = model.env.render(
            logits.detach().cpu().numpy().reshape(-1), volume_contraint=True)
    return compliance, np.clip(design, 0.0, 1.0)


def crossover_time(losses, total_time, bar):
    """First (step, ~seconds) at which losses <= bar; step times approximated
    by spreading total_time uniformly over the recorded steps."""
    hits = np.flatnonzero(np.asarray(losses) <= bar)
    if hits.size == 0:
        return None, None
    step = int(hits[0])
    return step, total_time * (step + 1) / len(losses)


def save_design_png(path, items, suptitle):
    """items: list of (title, design 2D array)."""
    fig, axes = plt.subplots(1, len(items), figsize=(5.5 * len(items), 3.4))
    if len(items) == 1:
        axes = [axes]
    for ax, (label, design) in zip(axes, items):
        ax.imshow(1.0 - design, cmap="gray", vmin=0.0, vmax=1.0)
        ax.set_title(label, fontsize=10)
        ax.axis("off")
    fig.suptitle(suptitle, fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_curve_png(path, scratch, finetune, zeroshot_compliance, cross_t, title):
    """scratch/finetune: (losses, total_time) tuples."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for label, (losses, total) in (("from scratch", scratch),
                                    ("fine-tuned transfer", finetune)):
        times = np.linspace(total / len(losses), total, len(losses))
        ax.plot(times, losses, label=label)
    ax.axhline(np.nanmin(scratch[0]), color="black", linestyle=":", lw=1,
               label="scratch final compliance")
    ax.axhline(zeroshot_compliance, color="gray", linestyle="--", lw=1,
               label="zero-shot (no training)")
    if cross_t is not None:
        ax.axvline(cross_t, color="red", linestyle="--", lw=1,
                   label=f"crossover (~{cross_t:.1f}s)")
    ax.set_xlabel("Target-problem training time (s, approximate)")
    ax.set_ylabel("Compliance")
    ax.set_yscale("log")
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.4)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def run_category(category, cfg, args, out_dir):
    src_name = cfg["source"]
    kan_layers = tuple(int(s) for s in args.kan_layers.split(",") if s.strip())
    cat_dir = out_dir / category
    cat_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 70}\nCATEGORY: {category}  (source: {src_name})\n{'=' * 70}")

    # --- Pre-train the source once (the amortized cost) ---
    print(f"[pretrain] {src_name} for {args.pretrain_steps} steps...")
    src_model = build_model(src_name, args.seed, kan_layers, args.grid, args.k)
    t0 = time.time()
    pt.train_lbfgs(src_model, args.pretrain_steps, progress_every=25,
                   save_intermediate_designs=False)
    pretrain_time = time.time() - t0
    print(f"[pretrain] done in {pretrain_time:.2f}s")
    src_state = copy.deepcopy(src_model.kan.state_dict())

    records = []
    for tgt_name in cfg["targets"]:
        prob = problems.PROBLEMS_BY_NAME[tgt_name]
        print(f"\n--- target: {tgt_name} ({prob.width}x{prob.height}, "
              f"density={prob.density}) ---")

        print(f"[scratch]  {args.steps} steps...")
        scratch_model = build_model(tgt_name, args.seed + 1, kan_layers,
                                     args.grid, args.k)
        s_comp, s_design, s_losses, s_time = train_timed(scratch_model, args.steps)
        print(f"[scratch]  compliance={s_comp:.4f} in {s_time:.2f}s")

        zeroshot_model = build_model(tgt_name, args.seed + 1, kan_layers,
                                      args.grid, args.k)
        zeroshot_model.kan.load_state_dict(copy.deepcopy(src_state))
        z_comp, z_design = zero_shot_eval(zeroshot_model)
        print(f"[zeroshot] compliance={z_comp:.4f} (0s)")

        print(f"[transfer] fine-tuning {args.steps} steps...")
        finetune_model = build_model(tgt_name, args.seed + 1, kan_layers,
                                      args.grid, args.k)
        finetune_model.kan.load_state_dict(copy.deepcopy(src_state))
        f_comp, f_design, f_losses, f_time = train_timed(finetune_model, args.steps)
        cross_step, cross_t = crossover_time(f_losses, f_time, s_comp)
        effective_time = cross_t if cross_t is not None else f_time
        print(f"[transfer] compliance={f_comp:.4f} in {f_time:.2f}s"
              + (f", matched scratch at step {cross_step + 1} (~{cross_t:.2f}s)"
                 if cross_step is not None else ", never matched scratch"))

        save_design_png(
            cat_dir / f"{tgt_name}_designs.png",
            [(f"from scratch\ncompliance={s_comp:.2f}, {s_time:.1f}s", s_design),
             (f"zero-shot transfer\ncompliance={z_comp:.2f}, 0s", z_design),
             (f"fine-tuned transfer\ncompliance={f_comp:.2f}, {f_time:.1f}s", f_design)],
            suptitle=tgt_name,
        )
        save_curve_png(
            cat_dir / f"{tgt_name}_curves.png",
            (s_losses, s_time), (f_losses, f_time), z_comp, cross_t,
            title=f"{tgt_name}: scratch vs transferred KAN",
        )

        records.append({
            "target": tgt_name,
            "width": prob.width, "height": prob.height, "density": prob.density,
            "scratch_compliance": s_comp, "scratch_time_s": s_time,
            "zeroshot_compliance": z_comp,
            "finetune_compliance": f_comp, "finetune_time_s": f_time,
            "crossover_step": cross_step,
            "crossover_time_s": cross_t,
            "effective_transfer_time_s": effective_time,
        })

    total_scratch = sum(r["scratch_time_s"] for r in records)
    total_transfer = pretrain_time + sum(r["effective_transfer_time_s"]
                                          for r in records)
    return {
        "category": category,
        "source": src_name,
        "pretrain_steps": args.pretrain_steps,
        "pretrain_time_s": pretrain_time,
        "targets": records,
        "total_scratch_time_s": total_scratch,
        "total_transfer_time_s": total_transfer,
        "time_saved_s": total_scratch - total_transfer,
        "time_saved_pct": 100.0 * (total_scratch - total_transfer) / total_scratch
                           if total_scratch > 0 else float("nan"),
    }


def write_summary(results, args, path):
    lines = ["# KAN Reusability: Amortization Experiment\n",
             f"KAN: layers=({args.kan_layers}), grid={args.grid}, k={args.k}; "
             f"pretrain={args.pretrain_steps} steps, target budget={args.steps} "
             f"steps, seed={args.seed}\n"]
    for res in results:
        lines.append(f"\n## {res['category']}  (source: {res['source']}, "
                     f"pre-trained in {res['pretrain_time_s']:.1f}s)\n")
        lines.append("| Target | Scratch compliance | Scratch time | Zero-shot | "
                     "Fine-tuned | Crossover | Effective transfer time |")
        lines.append("|---|---|---|---|---|---|---|")
        for r in res["targets"]:
            cross = (f"step {r['crossover_step'] + 1} (~{r['crossover_time_s']:.1f}s)"
                     if r["crossover_step"] is not None else "never")
            lines.append(
                f"| {r['target']} | {r['scratch_compliance']:.4f} | "
                f"{r['scratch_time_s']:.1f}s | {r['zeroshot_compliance']:.4f} | "
                f"{r['finetune_compliance']:.4f} | {cross} | "
                f"{r['effective_transfer_time_s']:.1f}s |")
        lines.append(
            f"\n**Amortized total**: scratch {res['total_scratch_time_s']:.1f}s "
            f"vs transfer {res['total_transfer_time_s']:.1f}s (incl. pre-train) "
            f"=> **{res['time_saved_s']:+.1f}s ({res['time_saved_pct']:+.1f}%)**\n")
    path.write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--categories", nargs="+",
                        default=list(AMORTIZATION_SETS),
                        choices=list(AMORTIZATION_SETS))
    parser.add_argument("--pretrain_steps", type=int, default=100)
    parser.add_argument("--steps", type=int, default=300,
                        help="L-BFGS budget per target run (scratch and fine-tune)")
    parser.add_argument("--kan_layers", type=str, default="16,16")
    parser.add_argument("--grid", type=int, default=8)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default="amortization_results")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    t_start = time.time()
    results = [run_category(c, AMORTIZATION_SETS[c], args, out_dir)
               for c in args.categories]

    with open(out_dir / "results.json", "w") as f:
        json.dump({"args": vars(args), "results": results}, f, indent=2)
    write_summary(results, args, out_dir / "summary.md")

    print(f"\n{'=' * 70}\nAMORTIZATION SUMMARY\n{'=' * 70}")
    for res in results:
        print(f"{res['category']:<28} scratch {res['total_scratch_time_s']:8.1f}s | "
              f"transfer {res['total_transfer_time_s']:8.1f}s (incl. pretrain) | "
              f"saved {res['time_saved_s']:+8.1f}s ({res['time_saved_pct']:+.1f}%)")
    print(f"\nTotal experiment wall-clock: {time.time() - t_start:.1f}s")
    print(f"Outputs written to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
