"""Phase 3 vs Phase 5 head-to-head evaluation.

Evaluates both checkpoints on both validation sets — a 2×2 grid:

                       Phase 3 (ICU-trained)     Phase 5 (Aurora-trained)
    MIMIC val          in-domain                 cross-domain
    Aurora-BP val      cross-domain              in-domain

The headline cell is **Phase 5 on Aurora val** (does in-domain training work?),
with **Phase 3 on Aurora val** as the empirical floor — an actual measurement of
the domain-shift loss instead of the n=1 CentrePoint projection from the
2026-05-27 session.  The Phase 5 × MIMIC val cell tells us whether Phase 5
generalizes back to ICU (informative for the paper, not for deployment).

Re-uses the inference + preprocessing helpers from path_a_radha.evaluate so the
per-subject metrics are bit-for-bit identical to what evaluate.py produces in
isolation — only the aggregation across cells is new.

Outputs (all under ``--out-dir``, default ``benchmark/results/phase3_vs_phase5/``):

  - comparison_table.json        — every metric in pooled_summary.json shape, per cell
  - comparison_table.md          — markdown tables ready to paste into a report
  - per_subject_metrics.json     — per-subject Pearson r etc. for each cell
  - pearson_r_histograms.png     — 4 panels: SBP/DBP × MIMIC/Aurora val,
                                   each overlaying Phase 3 and Phase 5 per-subject r
  - delta_summary.md             — short, opinionated "what changed" writeup

Usage (defaults assume the standard repo layout):

    python -m path_a_radha.evaluate_phase3_vs_phase5

To skip cells whose inputs aren't ready yet:

    python -m path_a_radha.evaluate_phase3_vs_phase5 \\
        --skip-aurora        # only the two MIMIC-val cells
    python -m path_a_radha.evaluate_phase3_vs_phase5 \\
        --skip-phase5        # only Phase 3 on both domains
"""
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from common.eval import bhs_grade, bland_altman, within_person_correlations
from path_a_radha.evaluate import (
    load_model_and_preprocessing,
    predict_with_mc_dropout,
    preprocess_features,
    trend_direction_agreement,
)
from path_a_radha.train import _subject_split

logger = logging.getLogger(__name__)


# ───────────────────────────────────────────────────────────────────────────
# Cell-level eval (mirrors evaluate.py's main() body, refactored as a function)
# ───────────────────────────────────────────────────────────────────────────

@dataclass
class CellResult:
    """One (model, val-set) combination's evaluation result."""
    model_label: str
    data_label: str
    pooled: dict = field(default_factory=dict)         # same keys as pooled_summary.json
    per_subject: dict[str, dict] = field(default_factory=dict)
    pearson_sbp: np.ndarray = field(default_factory=lambda: np.array([]))
    pearson_dbp: np.ndarray = field(default_factory=lambda: np.array([]))


def evaluate_on_val(
    checkpoint: Path,
    data_dir: Path,
    model_label: str,
    data_label: str,
    val_fraction: float = 0.20,
    seed: int = 42,
    sqi_threshold: float = 0.3,
    min_windows: int = 3,
    n_mc: int = 30,
    device: str = "cpu",
) -> CellResult:
    """Run a single (checkpoint, data_dir) evaluation cell.

    Reproduces the subject-disjoint val split with seed=42 (matches train.py),
    so 'val set' here means the held-out 20% of subjects in data_dir.  When
    checkpoint was trained on a *different* data_dir entirely (e.g. Phase 3
    evaluated against the Aurora .npz files), all subjects are effectively
    held-out — the val-split slice is still used for an apples-to-apples
    n with the in-domain cell.
    """
    from scipy.stats import pearsonr, spearmanr

    cell = CellResult(model_label=model_label, data_label=data_label)
    npz_paths = sorted(data_dir.glob("*.npz"))
    if not npz_paths:
        raise FileNotFoundError(f"No .npz files in {data_dir}")
    _, val_paths = _subject_split(npz_paths, val_fraction, seed)
    logger.info(
        "[%s × %s] %d total day-files → %d val day-files",
        model_label, data_label, len(npz_paths), len(val_paths),
    )

    stats = load_model_and_preprocessing(checkpoint, device=device)
    logger.info(
        "  loaded ckpt %s (epoch=%s, val_loss=%.2f, n_features=%d)",
        checkpoint, stats["epoch"], stats["val_loss"], stats["feat_mean"].shape[0],
    )

    all_pred_sbp: list[float] = []
    all_pred_dbp: list[float] = []
    all_ref_sbp:  list[float] = []
    all_ref_dbp:  list[float] = []
    all_pred_sbp_std: list[float] = []
    all_pred_dbp_std: list[float] = []
    all_pid: list[str] = []
    per_subject: dict[str, dict] = {}

    for path in val_paths:
        data = np.load(path, allow_pickle=True)
        features = data["features"].astype(np.float32)
        targets  = data["targets"].astype(np.float32)
        sqi      = data["sqi"].astype(np.float32) if "sqi" in data else np.ones(len(features))
        pid = (
            str(data["participant_id"])
            if "participant_id" in data
            else path.stem.rsplit("_", 1)[0]
        )

        mask = (sqi >= sqi_threshold) & np.all(np.isfinite(targets), axis=1)
        features = features[mask]
        targets  = targets[mask]
        if len(features) < min_windows:
            continue

        # Relativize per record (matches ParticipantDayDataset)
        targets = targets - targets.mean(axis=0, keepdims=True)

        feats_normed = preprocess_features(features, stats)
        pred_mean, pred_std = predict_with_mc_dropout(
            stats["model"], feats_normed, n_mc=n_mc, device=device,
        )

        try:
            pr_sbp, _ = pearsonr(pred_mean[:, 0], targets[:, 0])
            sr_sbp, _ = spearmanr(pred_mean[:, 0], targets[:, 0])
            pr_dbp, _ = pearsonr(pred_mean[:, 1], targets[:, 1])
            sr_dbp, _ = spearmanr(pred_mean[:, 1], targets[:, 1])
        except Exception:
            pr_sbp = sr_sbp = pr_dbp = sr_dbp = float("nan")

        per_subject[pid] = {
            "n_windows":           int(len(features)),
            "pearson_sbp":         float(pr_sbp),
            "spearman_sbp":        float(sr_sbp),
            "rmse_sbp":            float(np.sqrt(np.mean((pred_mean[:, 0] - targets[:, 0]) ** 2))),
            "trend_agreement_sbp": trend_direction_agreement(pred_mean[:, 0], targets[:, 0]),
            "pearson_dbp":         float(pr_dbp),
            "spearman_dbp":        float(sr_dbp),
            "rmse_dbp":            float(np.sqrt(np.mean((pred_mean[:, 1] - targets[:, 1]) ** 2))),
            "trend_agreement_dbp": trend_direction_agreement(pred_mean[:, 1], targets[:, 1]),
            "mean_pred_uncertainty_sbp": float(pred_std[:, 0].mean()),
            "mean_pred_uncertainty_dbp": float(pred_std[:, 1].mean()),
        }

        all_pred_sbp.extend(pred_mean[:, 0]); all_pred_dbp.extend(pred_mean[:, 1])
        all_ref_sbp.extend(targets[:, 0]);   all_ref_dbp.extend(targets[:, 1])
        all_pred_sbp_std.extend(pred_std[:, 0]); all_pred_dbp_std.extend(pred_std[:, 1])
        all_pid.extend([pid] * len(features))

    pred_sbp = np.asarray(all_pred_sbp); ref_sbp = np.asarray(all_ref_sbp)
    pred_dbp = np.asarray(all_pred_dbp); ref_dbp = np.asarray(all_ref_dbp)
    pids_arr = np.asarray(all_pid)
    if len(pred_sbp) == 0:
        logger.warning("[%s × %s] No usable windows passed QC.", model_label, data_label)
        cell.per_subject = per_subject
        return cell

    pearson_sbp_arr, spearman_sbp_arr = within_person_correlations(pred_sbp, ref_sbp, pids_arr)
    pearson_dbp_arr, spearman_dbp_arr = within_person_correlations(pred_dbp, ref_dbp, pids_arr)
    err_sbp = pred_sbp - ref_sbp
    err_dbp = pred_dbp - ref_dbp
    ba_sbp = bland_altman(pred_sbp, ref_sbp)
    ba_dbp = bland_altman(pred_dbp, ref_dbp)

    def _median(arr: np.ndarray) -> float:
        return float(np.median(arr)) if len(arr) else float("nan")

    def _pct(arr: np.ndarray, q: int) -> float:
        return float(np.percentile(arr, q)) if len(arr) else float("nan")

    trend_sbp_vals = [m["trend_agreement_sbp"] for m in per_subject.values()
                      if np.isfinite(m["trend_agreement_sbp"])]
    trend_dbp_vals = [m["trend_agreement_dbp"] for m in per_subject.values()
                      if np.isfinite(m["trend_agreement_dbp"])]

    cell.pooled = {
        "model":                  model_label,
        "data":                   data_label,
        "checkpoint":             str(checkpoint),
        "data_dir":               str(data_dir),
        "n_val_records":          len(per_subject),
        "n_val_windows":          int(len(pred_sbp)),
        "best_epoch":             stats["epoch"],
        "val_loss":               float(stats["val_loss"]),
        "pearson_sbp_median":     _median(pearson_sbp_arr),
        "pearson_sbp_q25":        _pct(pearson_sbp_arr, 25),
        "pearson_sbp_q75":        _pct(pearson_sbp_arr, 75),
        "spearman_sbp_median":    _median(spearman_sbp_arr),
        "pearson_dbp_median":     _median(pearson_dbp_arr),
        "pearson_dbp_q25":        _pct(pearson_dbp_arr, 25),
        "pearson_dbp_q75":        _pct(pearson_dbp_arr, 75),
        "spearman_dbp_median":    _median(spearman_dbp_arr),
        "trend_agree_sbp_median": float(np.median(trend_sbp_vals)) if trend_sbp_vals else float("nan"),
        "trend_agree_dbp_median": float(np.median(trend_dbp_vals)) if trend_dbp_vals else float("nan"),
        "sbp_mae":   float(np.mean(np.abs(err_sbp))),
        "sbp_rmse":  float(np.sqrt(np.mean(err_sbp ** 2))),
        "sbp_me":    float(np.mean(err_sbp)),
        "sbp_ba_mean": ba_sbp[0], "sbp_ba_loa_upper": ba_sbp[1], "sbp_ba_loa_lower": ba_sbp[2],
        "sbp_bhs":   bhs_grade(err_sbp),
        "dbp_mae":   float(np.mean(np.abs(err_dbp))),
        "dbp_rmse":  float(np.sqrt(np.mean(err_dbp ** 2))),
        "dbp_me":    float(np.mean(err_dbp)),
        "dbp_ba_mean": ba_dbp[0], "dbp_ba_loa_upper": ba_dbp[1], "dbp_ba_loa_lower": ba_dbp[2],
        "dbp_bhs":   bhs_grade(err_dbp),
        "mean_uncertainty_sbp": float(np.mean(all_pred_sbp_std)),
        "mean_uncertainty_dbp": float(np.mean(all_pred_dbp_std)),
        # Predicted dynamic range — the "predicts-near-mean" tell.  A flat model
        # has predicted_std ≪ reference_std even when the reference moves.
        "pred_sbp_std":          float(pred_sbp.std()),
        "ref_sbp_std":           float(ref_sbp.std()),
        "pred_dbp_std":          float(pred_dbp.std()),
        "ref_dbp_std":           float(ref_dbp.std()),
        "range_compression_sbp": float(pred_sbp.std() / ref_sbp.std()) if ref_sbp.std() > 0 else float("nan"),
        "range_compression_dbp": float(pred_dbp.std() / ref_dbp.std()) if ref_dbp.std() > 0 else float("nan"),
    }
    cell.per_subject = per_subject
    cell.pearson_sbp = pearson_sbp_arr
    cell.pearson_dbp = pearson_dbp_arr
    return cell


# ───────────────────────────────────────────────────────────────────────────
# Aggregation: markdown + JSON + figure
# ───────────────────────────────────────────────────────────────────────────

def _fmt(val: float, fmt: str = "{:+.3f}") -> str:
    if val is None or (isinstance(val, float) and not np.isfinite(val)):
        return "—"
    return fmt.format(val)


def write_comparison_table_md(cells: list[CellResult], out_path: Path) -> None:
    """Write a markdown report with the headline grid + full numbers."""
    lines: list[str] = []
    lines.append("# Phase 3 vs Phase 5 — Head-to-Head Evaluation\n")
    lines.append("Within-person trend correlation is the primary metric per CLAUDE.md ")
    lines.append("(Garrett Ash's reframe). Pooled MAE/RMSE/BHS are reported for completeness; ")
    lines.append("they're easy to game with a flat predictor, hence the `range_compression` column.\n\n")

    # 2x2 headline: SBP within-person Pearson r median
    by_key: dict[tuple[str, str], CellResult] = {(c.model_label, c.data_label): c for c in cells}
    models = sorted({c.model_label for c in cells})
    domains = sorted({c.data_label for c in cells})

    lines.append("## Headline: within-person SBP Pearson r (median [Q25, Q75])\n\n")
    header = "| Eval set / Model | " + " | ".join(models) + " |"
    sep    = "|" + "---|" * (len(models) + 1)
    lines.append(header)
    lines.append(sep)
    for d in domains:
        row = [f"**{d}**"]
        for m in models:
            c = by_key.get((m, d))
            if c is None or not c.pooled:
                row.append("—")
            else:
                p = c.pooled
                row.append(f"{_fmt(p['pearson_sbp_median'])} "
                           f"[{_fmt(p['pearson_sbp_q25'])}, {_fmt(p['pearson_sbp_q75'])}] "
                           f"(n={p['n_val_records']})")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    lines.append("## Headline: within-person DBP Pearson r (median [Q25, Q75])\n\n")
    lines.append(header)
    lines.append(sep)
    for d in domains:
        row = [f"**{d}**"]
        for m in models:
            c = by_key.get((m, d))
            if c is None or not c.pooled:
                row.append("—")
            else:
                p = c.pooled
                row.append(f"{_fmt(p['pearson_dbp_median'])} "
                           f"[{_fmt(p['pearson_dbp_q25'])}, {_fmt(p['pearson_dbp_q75'])}] "
                           f"(n={p['n_val_records']})")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    # Range compression — the predicts-near-mean tell
    lines.append("## Range compression (predicts-near-mean indicator)\n\n")
    lines.append("`std(pred) / std(reference)` per cell. A healthy tracker is near 1.0; ")
    lines.append("a flat 'predicts-near-mean' collapse is ≪ 1.0 (e.g. 0.1).\n\n")
    lines.append("| Eval set | Model | SBP std (pred / ref) | DBP std (pred / ref) | compression SBP | compression DBP |")
    lines.append("|---|---|---|---|---|---|")
    for d in domains:
        for m in models:
            c = by_key.get((m, d))
            if c is None or not c.pooled:
                continue
            p = c.pooled
            lines.append(
                f"| {d} | {m} | "
                f"{_fmt(p['pred_sbp_std'], '{:.2f}')} / {_fmt(p['ref_sbp_std'], '{:.2f}')} | "
                f"{_fmt(p['pred_dbp_std'], '{:.2f}')} / {_fmt(p['ref_dbp_std'], '{:.2f}')} | "
                f"{_fmt(p['range_compression_sbp'], '{:.2f}')} | "
                f"{_fmt(p['range_compression_dbp'], '{:.2f}')} |"
            )
    lines.append("")

    # Full pooled metrics
    lines.append("## Pooled population metrics\n\n")
    lines.append("| Eval set | Model | n rec | n win | SBP MAE | SBP RMSE | SBP BHS | DBP MAE | DBP RMSE | DBP BHS |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for d in domains:
        for m in models:
            c = by_key.get((m, d))
            if c is None or not c.pooled:
                continue
            p = c.pooled
            lines.append(
                f"| {d} | {m} | {p['n_val_records']} | {p['n_val_windows']} | "
                f"{_fmt(p['sbp_mae'], '{:.2f}')} | {_fmt(p['sbp_rmse'], '{:.2f}')} | {p['sbp_bhs']} | "
                f"{_fmt(p['dbp_mae'], '{:.2f}')} | {_fmt(p['dbp_rmse'], '{:.2f}')} | {p['dbp_bhs']} |"
            )
    lines.append("")

    # Trend-direction agreement (subject-median)
    lines.append("## Trend-direction agreement (subject-median fraction; 0.5 = chance)\n\n")
    lines.append("| Eval set | Model | SBP | DBP |")
    lines.append("|---|---|---|---|")
    for d in domains:
        for m in models:
            c = by_key.get((m, d))
            if c is None or not c.pooled:
                continue
            p = c.pooled
            lines.append(
                f"| {d} | {m} | "
                f"{_fmt(p['trend_agree_sbp_median'], '{:.3f}')} | "
                f"{_fmt(p['trend_agree_dbp_median'], '{:.3f}')} |"
            )
    lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote %s", out_path)


def write_delta_summary_md(cells: list[CellResult], out_path: Path) -> None:
    """Short opinionated 'what changed' writeup, focusing on the headline deltas."""
    by_key = {(c.model_label, c.data_label): c for c in cells}

    def cell_pearson(model: str, domain: str, channel: str) -> float | None:
        c = by_key.get((model, domain))
        if c is None or not c.pooled:
            return None
        return c.pooled.get(f"pearson_{channel}_median")

    def cell_compression(model: str, domain: str, channel: str) -> float | None:
        c = by_key.get((model, domain))
        if c is None or not c.pooled:
            return None
        return c.pooled.get(f"range_compression_{channel}")

    def delta(a: float | None, b: float | None) -> str:
        if a is None or b is None or not (np.isfinite(a) and np.isfinite(b)):
            return "—"
        return f"{a - b:+.3f}"

    lines: list[str] = []
    lines.append("# Phase 3 → Phase 5 deltas\n")
    lines.append("Δ = Phase 5 − Phase 3 on the same val set; positive Δ on Pearson r ")
    lines.append("means Phase 5 tracks BP better than Phase 3.\n")

    p3_aur_sbp = cell_pearson("Phase 3", "Aurora val", "sbp")
    p5_aur_sbp = cell_pearson("Phase 5", "Aurora val", "sbp")
    p3_aur_dbp = cell_pearson("Phase 3", "Aurora val", "dbp")
    p5_aur_dbp = cell_pearson("Phase 5", "Aurora val", "dbp")
    p3_mim_sbp = cell_pearson("Phase 3", "MIMIC val", "sbp")
    p5_mim_sbp = cell_pearson("Phase 5", "MIMIC val", "sbp")

    lines.append("## On Aurora val (the deployment-relevant domain)\n")
    lines.append(f"- SBP Pearson r: Phase 3 = {_fmt(p3_aur_sbp)}, "
                 f"Phase 5 = {_fmt(p5_aur_sbp)}  →  **Δ = {delta(p5_aur_sbp, p3_aur_sbp)}**")
    lines.append(f"- DBP Pearson r: Phase 3 = {_fmt(p3_aur_dbp)}, "
                 f"Phase 5 = {_fmt(p5_aur_dbp)}  →  **Δ = {delta(p5_aur_dbp, p3_aur_dbp)}**")
    lines.append(f"- Range compression SBP: Phase 3 = {_fmt(cell_compression('Phase 3', 'Aurora val', 'sbp'), '{:.2f}')}, "
                 f"Phase 5 = {_fmt(cell_compression('Phase 5', 'Aurora val', 'sbp'), '{:.2f}')}  "
                 "(closer to 1.0 = less flat)")
    lines.append("")

    lines.append("## On MIMIC val (sanity check — does Phase 5 also work on ICU?)\n")
    lines.append(f"- SBP Pearson r: Phase 3 = {_fmt(p3_mim_sbp)} (the published baseline), "
                 f"Phase 5 = {_fmt(p5_mim_sbp)}  →  Δ = {delta(p5_mim_sbp, p3_mim_sbp)}")
    lines.append("  - Phase 5 was never trained on ICU patients; any positive r here is generalization.")
    lines.append("")

    lines.append("## Interpretation guide\n")
    lines.append("- **Δ(Aurora SBP) ≥ +0.15** → in-domain retraining works; Phase 5 is the new deployment candidate.")
    lines.append("- **Δ(Aurora SBP) in [0, +0.15)** → modest gain — fine-tuning per participant may still be needed.")
    lines.append("- **Δ(Aurora SBP) ≤ 0** → Phase 5 didn't transfer either; the wrist-PPG-to-BP ceiling is hit ")
    lines.append("  regardless of domain (consistent with the 'honest ceiling' caveat in the session summary).")
    lines.append("- Watch the **range compression** column independently: an r gain with compression still ≪ 1.0 ")
    lines.append("  means the model learned ordering but not magnitudes — useful for trend reports, not for ")
    lines.append("  absolute BP claims.")
    lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote %s", out_path)


def plot_pearson_histograms(cells: list[CellResult], out_path: Path) -> None:
    """4-panel figure: SBP/DBP × MIMIC/Aurora, each overlaying both models' per-subject r."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    by_key = {(c.model_label, c.data_label): c for c in cells}
    domains = sorted({c.data_label for c in cells})

    fig, axes = plt.subplots(2, len(domains), figsize=(5.5 * len(domains), 8), squeeze=False)
    channels = [("SBP", "pearson_sbp"), ("DBP", "pearson_dbp")]
    bins = np.linspace(-1.0, 1.0, 21)

    for col, d in enumerate(domains):
        for row, (chan_label, attr) in enumerate(channels):
            ax = axes[row][col]
            for m, color in [("Phase 3", "C0"), ("Phase 5", "C3")]:
                c = by_key.get((m, d))
                if c is None or len(getattr(c, attr)) == 0:
                    continue
                arr = getattr(c, attr)
                med = float(np.median(arr))
                ax.hist(arr, bins=bins, alpha=0.45, color=color, edgecolor=color,
                        label=f"{m}  (median {med:+.2f}, n={len(arr)})")
                ax.axvline(med, color=color, lw=2, ls="--")
            ax.axvline(0.0, color="k", lw=0.8, alpha=0.4)
            ax.set_xlabel(f"per-subject Pearson r ({chan_label})")
            ax.set_ylabel("# subjects")
            ax.set_title(f"{chan_label} • {d}")
            ax.set_xlim(-1.0, 1.0)
            ax.legend(loc="upper left", fontsize=9)
            ax.grid(alpha=0.3)

    fig.suptitle("Phase 3 vs Phase 5 — within-person Pearson r by evaluation set", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    logger.info("Wrote %s", out_path)


# ───────────────────────────────────────────────────────────────────────────
# CLI
# ───────────────────────────────────────────────────────────────────────────

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="Phase 3 vs Phase 5 head-to-head eval")
    ap.add_argument("--phase3-checkpoint", type=Path,
                    default=Path("benchmark/results/path_a_first_run/best_model.pt"))
    ap.add_argument("--phase5-checkpoint", type=Path,
                    default=Path("benchmark/results/path_a_phase5/best_model.pt"))
    ap.add_argument("--mimic-data-dir",  type=Path, default=Path("data/processed/path_a"))
    ap.add_argument("--aurora-data-dir", type=Path, default=Path("data/processed/path_a_aurora"))
    ap.add_argument("--out-dir", type=Path, default=Path("benchmark/results/phase3_vs_phase5"))
    ap.add_argument("--skip-aurora",  action="store_true",
                    help="omit both Aurora-val cells (e.g., adapter hasn't been run yet)")
    ap.add_argument("--skip-mimic",   action="store_true",
                    help="omit both MIMIC-val cells")
    ap.add_argument("--skip-phase5",  action="store_true",
                    help="omit Phase 5 cells (e.g., training not finished)")
    ap.add_argument("--skip-phase3",  action="store_true",
                    help="omit Phase 3 cells")
    ap.add_argument("--n-mc", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val-fraction", type=float, default=0.20)
    ap.add_argument("--sqi-threshold", type=float, default=0.3)
    ap.add_argument("--min-windows", type=int, default=3)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Build the list of cells to attempt — skip flags + existence checks
    candidates: list[tuple[str, Path, str, Path]] = []
    if not args.skip_phase3:
        if not args.skip_mimic:
            candidates.append(("Phase 3", args.phase3_checkpoint, "MIMIC val",  args.mimic_data_dir))
        if not args.skip_aurora:
            candidates.append(("Phase 3", args.phase3_checkpoint, "Aurora val", args.aurora_data_dir))
    if not args.skip_phase5:
        if not args.skip_mimic:
            candidates.append(("Phase 5", args.phase5_checkpoint, "MIMIC val",  args.mimic_data_dir))
        if not args.skip_aurora:
            candidates.append(("Phase 5", args.phase5_checkpoint, "Aurora val", args.aurora_data_dir))

    cells: list[CellResult] = []
    for model_label, ckpt, data_label, data_dir in candidates:
        if not ckpt.exists():
            logger.warning("[%s × %s] checkpoint not found at %s — skipping",
                           model_label, data_label, ckpt)
            continue
        if not data_dir.exists() or not any(data_dir.glob("*.npz")):
            logger.warning("[%s × %s] data dir empty or missing: %s — skipping",
                           model_label, data_label, data_dir)
            continue
        try:
            cell = evaluate_on_val(
                checkpoint=ckpt,
                data_dir=data_dir,
                model_label=model_label,
                data_label=data_label,
                val_fraction=args.val_fraction,
                seed=args.seed,
                sqi_threshold=args.sqi_threshold,
                min_windows=args.min_windows,
                n_mc=args.n_mc,
            )
            cells.append(cell)
        except Exception as exc:
            logger.exception("[%s × %s] failed: %s", model_label, data_label, exc)

    if not cells:
        raise SystemExit("No cells were evaluated — nothing to compare.")

    # ── Persist results ──
    pooled_payload = {f"{c.model_label} | {c.data_label}": c.pooled for c in cells}
    (args.out_dir / "comparison_table.json").write_text(
        json.dumps(pooled_payload, indent=2), encoding="utf-8",
    )
    per_subject_payload = {
        f"{c.model_label} | {c.data_label}": c.per_subject for c in cells
    }
    (args.out_dir / "per_subject_metrics.json").write_text(
        json.dumps(per_subject_payload, indent=2), encoding="utf-8",
    )

    write_comparison_table_md(cells, args.out_dir / "comparison_table.md")
    write_delta_summary_md(cells,    args.out_dir / "delta_summary.md")
    plot_pearson_histograms(cells,   args.out_dir / "pearson_r_histograms.png")

    # ── Print a concise console summary ──
    print()
    print("=" * 72)
    print("PHASE 3 vs PHASE 5 — HEAD-TO-HEAD")
    print("=" * 72)
    for c in cells:
        if not c.pooled:
            print(f"  [{c.model_label} × {c.data_label}]  no usable val data")
            continue
        p = c.pooled
        print(f"  [{c.model_label:8s} × {c.data_label:11s}]  "
              f"SBP r = {p['pearson_sbp_median']:+.3f} "
              f"[Q25 {p['pearson_sbp_q25']:+.3f}, Q75 {p['pearson_sbp_q75']:+.3f}]  "
              f"DBP r = {p['pearson_dbp_median']:+.3f}  "
              f"SBP MAE = {p['sbp_mae']:5.2f}  "
              f"BHS = {p['sbp_bhs']}/{p['dbp_bhs']}  "
              f"compression SBP = {p['range_compression_sbp']:.2f}  "
              f"n_rec = {p['n_val_records']:3d}")
    print()
    print(f"  Artifacts: {args.out_dir}")


if __name__ == "__main__":
    main()
