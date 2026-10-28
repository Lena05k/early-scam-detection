#!/usr/bin/env python3
"""Групповой стратифицированный сплит train/val/test корпуса (шаг 3).

Единица распределения — scenario_group; стратификация язык × is_scam ×
бакет онсета; рандомизированный жадный поиск с минимизацией суммарной JSD
(>= 1000 проходов от master seed). Вписывает поле split в JSONL, пишет
независимый маппинг splits_v1.json, отчёт и фигуры для статьи.

Пример:
  python scripts/make_splits.py
  python scripts/make_splits.py --trials 2000 --no-figures

Перед записью делает резервную копию входного файла (<file>.bak_splits),
если её ещё нет.
"""
import argparse
import json
import pathlib
import shutil
import sys
from collections import Counter

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import yaml  # noqa: E402

from scam_onset.data import splits as sp  # noqa: E402

DEFAULT_IN = "data/corpus/corpus_v1.jsonl"
DEFAULT_MAP = "data/corpus/splits_v1.json"
DEFAULT_REPORT = "reports/splits_v1_report.json"
DEFAULT_FIGDIR = "reports/figures"
DEFAULT_CONFIG = "configs/default.yaml"

# --- палитра фигур (валидированная категориальная, light mode) ---
SPLIT_COLORS = {"train": "#2a78d6", "val": "#008300", "test": "#e87ba4"}
INK = "#0b0b0b"
MUTED = "#898781"
SECONDARY = "#52514e"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"
SEQ_BLUES = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, records):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_split_config(path):
    cfg = {}
    p = pathlib.Path(path)
    if p.exists():
        cfg = (yaml.safe_load(p.read_text(encoding="utf-8")) or {}).get("splits", {})
    return cfg


def composition_tables(records):
    """Таблицы состава по сплитам для отчёта."""
    tables = {"by_split": {}, "scenario_x_split": {}}
    for s in sp.SPLITS:
        rs = [r for r in records if r["split"] == s]
        tables["by_split"][s] = {
            "n": len(rs),
            "lang": dict(Counter(r["lang"] for r in rs)),
            "scam": sum(r["is_scam"] for r in rs),
            "benign": sum(1 for r in rs if r["is_scam"] == 0),
            "hard_neg": sum(r["hard_negative"] for r in rs),
            "onset_bucket": dict(Counter(sp.onset_bucket(r) for r in rs)),
            "mean_turns": round(sum(len(r["turns"]) for r in rs) / len(rs), 2),
        }
    for scen in sorted({r["scam_type"] for r in records}):
        tables["scenario_x_split"][scen] = {
            s: sum(1 for r in records if r["scam_type"] == scen and r["split"] == s)
            for s in sp.SPLITS}
    return tables


# ── фигуры ──────────────────────────────────────────────────────────────

def _style_axes(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c3c2b7")
    ax.tick_params(colors=MUTED, labelcolor=SECONDARY)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _save(fig, figdir, name):
    out = figdir / name
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    print(f"  фигура: {out}")
    import matplotlib.pyplot as plt
    plt.close(fig)
    return out


def make_figures(records, result, checks, figdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.lines import Line2D

    plt.rcParams.update({
        "font.family": "sans-serif", "text.color": INK,
        "axes.titlecolor": INK, "axes.labelcolor": SECONDARY,
        "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    })
    figdir = pathlib.Path(figdir)
    figdir.mkdir(parents=True, exist_ok=True)
    paths = []

    by_split = {s: [r for r in records if r["split"] == s] for s in sp.SPLITS}
    n_split = {s: len(by_split[s]) for s in sp.SPLITS}
    n_total = len(records)
    langs = ["ru", "en", "zh"]
    W = 0.24  # ширина бара в grouped-барах

    def grouped_bars(ax, cats, series, global_vals=None, unit="%"):
        """series: {split: [v...]} по cats; global_vals: [v...] маркером."""
        for si, s in enumerate(sp.SPLITS):
            x = [ci + (si - 1) * W for ci in range(len(cats))]
            bars = ax.bar(x, series[s], width=W - 0.02, color=SPLIT_COLORS[s],
                          label=s, zorder=2)
            for b, v in zip(bars, series[s]):
                ax.annotate(f"{v:.1f}" if unit == "%" else f"{v:.0f}",
                            (b.get_x() + b.get_width() / 2, b.get_height()),
                            ha="center", va="bottom", fontsize=7, color=SECONDARY)
        if global_vals is not None:
            for ci, v in enumerate(global_vals):
                ax.hlines(v, ci - 1.6 * W, ci + 1.6 * W, color=INK,
                          linestyle=(0, (4, 2)), linewidth=1.2, zorder=3)
        ax.set_xticks(range(len(cats)))
        ax.set_xticklabels(cats)
        _style_axes(ax)

    # a) баланс языков
    fig, ax = plt.subplots(figsize=(7, 4))
    series = {s: [100 * sum(1 for r in by_split[s] if r["lang"] == g) / n_split[s]
                  for g in langs] for s in sp.SPLITS}
    gvals = [100 * sum(1 for r in records if r["lang"] == g) / n_total for g in langs]
    grouped_bars(ax, langs, series, gvals)
    ax.set_ylabel("Share of split (%)")
    ax.set_title("Language balance across splits")
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Line2D([], [], color=INK, linestyle=(0, (4, 2)), linewidth=1.2))
    labels.append("global")
    ax.legend(handles, labels, frameon=False, ncol=4, loc="lower center",
              bbox_to_anchor=(0.5, -0.22))
    paths.append(_save(fig, figdir, "fig_split_lang_balance.png"))

    # b) баланс классов
    fig, ax = plt.subplots(figsize=(7, 4))
    cats = ["scam", "benign", "hard negative"]

    def class_shares(rs):
        n = len(rs)
        return [100 * sum(r["is_scam"] for r in rs) / n,
                100 * sum(1 for r in rs if r["is_scam"] == 0) / n,
                100 * sum(r["hard_negative"] for r in rs) / n]

    series = {s: class_shares(by_split[s]) for s in sp.SPLITS}
    grouped_bars(ax, cats, series, class_shares(records))
    ax.set_ylabel("Share of split (%)")
    ax.set_title("Class balance across splits (hard negative ⊂ benign)")
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Line2D([], [], color=INK, linestyle=(0, (4, 2)), linewidth=1.2))
    labels.append("global")
    ax.legend(handles, labels, frameon=False, ncol=4, loc="lower center",
              bbox_to_anchor=(0.5, -0.22))
    paths.append(_save(fig, figdir, "fig_split_class_balance.png"))

    # c) гистограммы онсета: абсолютный t* и нормированный t*/T
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    t_max = max(r["onset"] for r in records if r["is_scam"] == 1)
    ts = list(range(1, t_max + 1))
    for si, s in enumerate(sp.SPLITS):
        scam = [r for r in by_split[s] if r["is_scam"] == 1]
        cnt = Counter(r["onset"] for r in scam)
        vals = [100 * cnt.get(t, 0) / len(scam) for t in ts]
        x = [t + (si - 1) * W for t in ts]
        ax1.bar(x, vals, width=W - 0.02, color=SPLIT_COLORS[s], label=s, zorder=2)
    ax1.set_xticks(ts)
    ax1.set_xlabel("Onset turn $t^*$ (absolute)")
    ax1.set_ylabel("Share of scam dialogues (%)")
    ax1.set_title("Onset position by split")
    _style_axes(ax1)
    ax1.legend(frameon=False)

    bins = [i / 10 for i in range(11)]
    bw = 0.1 / 3
    for si, s in enumerate(sp.SPLITS):
        scam = [r for r in by_split[s] if r["is_scam"] == 1]
        rel = [r["onset"] / len(r["turns"]) for r in scam]
        hist = [0] * 10
        for v in rel:
            hist[min(int(v * 10), 9)] += 1
        vals = [100 * h / len(scam) for h in hist]
        x = [bins[i] + bw * (si + 0.5) for i in range(10)]
        ax2.bar(x, vals, width=bw * 0.9, color=SPLIT_COLORS[s], label=s, zorder=2)
    ax2.set_xticks(bins)
    ax2.set_xlabel("Normalized onset $t^*/T$")
    ax2.set_ylabel("Share of scam dialogues (%)")
    ax2.set_title("Normalized onset by split")
    _style_axes(ax2)
    ax2.legend(frameon=False)
    fig.tight_layout()
    paths.append(_save(fig, figdir, "fig_onset_hist_by_split.png"))

    # d) heatmap сценарий × сплит
    scens = sorted({r["scam_type"] for r in records})
    counts = [[sum(1 for r in by_split[s] if r["scam_type"] == sc)
               for s in sp.SPLITS] for sc in scens]
    shares = [[counts[i][j] / n_split[sp.SPLITS[j]] for j in range(3)]
              for i in range(len(scens))]
    cmap = LinearSegmentedColormap.from_list("blues", SEQ_BLUES)
    fig, ax = plt.subplots(figsize=(6, 8))
    vmax = max(max(row) for row in shares)
    im = ax.imshow(shares, cmap=cmap, aspect="auto", vmin=0, vmax=vmax)
    for i in range(len(scens)):
        for j in range(3):
            dark = shares[i][j] > 0.6 * vmax
            ax.text(j, i, str(counts[i][j]), ha="center", va="center",
                    fontsize=8, color="#ffffff" if dark else INK)
    ax.set_xticks(range(3))
    ax.set_xticklabels(sp.SPLITS)
    ax.set_yticks(range(len(scens)))
    ax.set_yticklabels(scens, fontsize=8)
    ax.set_title("Scenario coverage (label: dialogues; color: share of split)")
    ax.tick_params(colors=MUTED, labelcolor=SECONDARY)
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("Share of split", color=SECONDARY)
    cbar.ax.tick_params(colors=MUTED, labelcolor=SECONDARY)
    fig.tight_layout()
    paths.append(_save(fig, figdir, "fig_scenario_coverage.png"))

    # e) размеры групп по сплитам (лог-шкала)
    prof = sp.group_profiles(records)
    assignment = result["assignment"]
    sizes_by_split = {s: Counter() for s in sp.SPLITS}
    for g, p in prof.items():
        sizes_by_split[assignment[g]][p["size"]] += 1
    all_sizes = sorted({p["size"] for p in prof.values()})
    fig, ax = plt.subplots(figsize=(9, 4))
    for si, s in enumerate(sp.SPLITS):
        x = [i + (si - 1) * W for i in range(len(all_sizes))]
        vals = [sizes_by_split[s].get(sz, 0) for sz in all_sizes]
        bars = ax.bar(x, vals, width=W - 0.02, color=SPLIT_COLORS[s], label=s, zorder=2)
        for b, v in zip(bars, vals):
            if v:
                ax.annotate(str(v), (b.get_x() + b.get_width() / 2, b.get_height()),
                            ha="center", va="bottom", fontsize=7, color=SECONDARY)
    ax.set_yscale("log")
    ax.set_xticks(range(len(all_sizes)))
    ax.set_xticklabels(all_sizes)
    ax.set_xlabel("Scenario-group size (dialogues)")
    ax.set_ylabel("Number of groups (log)")
    ax.set_title("Scenario-group sizes by split")
    _style_axes(ax)
    ax.legend(frameon=False)
    paths.append(_save(fig, figdir, "fig_group_sizes.png"))

    # f) повкладовая JSD по стратам
    strata_split = {s: Counter() for s in sp.SPLITS}
    for r in records:
        strata_split[r["split"]][sp.stratum(r)] += 1
    dists = {s: {k: v / n_split[s] for k, v in strata_split[s].items()}
             for s in sp.SPLITS}
    keys = sorted({k for s in sp.SPLITS for k in strata_split[s]})
    c_tt = sp.jsd_contributions(dists["train"], dists["test"], keys)
    c_tv = sp.jsd_contributions(dists["train"], dists["val"], keys)
    fig, ax = plt.subplots(figsize=(10, 4))
    labels = [sp.stratum_label(k) for k in keys]
    x = list(range(len(keys)))
    ax.bar([i - 0.18 for i in x], [1e3 * c_tt[k] for k in keys], width=0.34,
           color="#2a78d6", label="train vs test", zorder=2)
    ax.bar([i + 0.18 for i in x], [1e3 * c_tv[k] for k in keys], width=0.34,
           color="#008300", label="train vs val", zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel(r"JSD contribution ($\times 10^{-3}$ bits)")
    ax.set_title("Per-stratum divergence between splits (lower is better)")
    _style_axes(ax)
    ax.legend(frameon=False)
    fig.tight_layout()
    paths.append(_save(fig, figdir, "fig_strata_divergence.png"))

    # g) длины диалогов по сплитам
    fig, ax = plt.subplots(figsize=(7, 4))
    data = [[len(r["turns"]) for r in by_split[s]] for s in sp.SPLITS]
    parts = ax.violinplot(data, positions=[0, 1, 2], showmedians=True, widths=0.7)
    for body, s in zip(parts["bodies"], sp.SPLITS):
        body.set_facecolor(SPLIT_COLORS[s])
        body.set_edgecolor(SPLIT_COLORS[s])
        body.set_alpha(0.75)
    for key in ("cbars", "cmins", "cmaxes", "cmedians"):
        parts[key].set_color(SECONDARY)
        parts[key].set_linewidth(1.2)
    for i, d in enumerate(data):
        med = sorted(d)[len(d) // 2]
        ax.annotate(f"median {med}", (i, max(d) + 0.3), ha="center",
                    fontsize=8, color=SECONDARY)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(sp.SPLITS)
    ax.set_ylabel("Dialogue length T (turns)")
    ax.set_title("Dialogue length distribution by split")
    _style_axes(ax)
    paths.append(_save(fig, figdir, "fig_turns_dist_by_split.png"))
    return paths


# ── main ────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="inp", default=DEFAULT_IN)
    ap.add_argument("--map", dest="map_out", default=DEFAULT_MAP)
    ap.add_argument("--report", default=DEFAULT_REPORT)
    ap.add_argument("--figdir", default=DEFAULT_FIGDIR)
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--trials", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--skip-leakage", action="store_true",
                    help="пропустить дорогую проверку утечки текста")
    args = ap.parse_args()

    cfg = load_split_config(args.config)
    fractions = cfg.get("fractions", sp.DEFAULT_FRACTIONS)
    master_seed = args.seed if args.seed is not None else cfg.get(
        "master_seed", sp.DEFAULT_MASTER_SEED)
    n_trials = args.trials if args.trials is not None else cfg.get(
        "n_trials", sp.DEFAULT_N_TRIALS)
    large_min = cfg.get("large_group_min_size", sp.LARGE_GROUP_MIN_SIZE)

    records = read_jsonl(args.inp)
    print(f"прочитано {len(records)} диалогов из {args.inp}")

    # резервная копия (не перезаписывать существующую)
    backup = pathlib.Path(args.inp + ".bak_splits")
    if not backup.exists():
        shutil.copy2(args.inp, backup)
        print(f"бэкап: {backup}")
    else:
        print(f"бэкап уже есть: {backup}")

    print(f"поиск сплита: {n_trials} проходов, master seed = {master_seed} …")
    result = sp.search_best_split(records, fractions=fractions,
                                  master_seed=master_seed, n_trials=n_trials,
                                  large_group_min_size=large_min)
    print(f"  валидных проходов: {result['n_valid_trials']}/{n_trials}; "
          f"лучший trial {result['best_trial']}, суммарная JSD = "
          f"{result['total_jsd']:.6f} bits")
    print("  крупные группы:", result["large"])

    # вписать split
    split_of = result["split_of"]
    for r in records:
        r["split"] = split_of[r["id"]]
    write_jsonl(args.inp, records)
    print(f"поле split вписано: {args.inp}")

    pathlib.Path(args.map_out).write_text(
        json.dumps(split_of, ensure_ascii=False, indent=0, sort_keys=True),
        encoding="utf-8")
    print(f"маппинг id → split: {args.map_out}")

    # валидация 3.5
    checks = sp.validate_split(records, fractions=fractions)
    if not args.skip_leakage:
        print("проверка утечки текста train↔test …")
        checks["text_leakage_train_test"] = sp.cross_split_leakage(
            records, "train", "test")
    for name, c in checks.items():
        print(f"  {'✅' if c['pass'] else '❌'} {name}")

    # отчёт
    strata_split = {s: Counter() for s in sp.SPLITS}
    n_split = Counter(r["split"] for r in records)
    for r in records:
        strata_split[r["split"]][sp.stratum(r)] += 1
    dists = {s: {k: v / n_split[s] for k, v in strata_split[s].items()}
             for s in sp.SPLITS}
    keys = sorted({k for s in sp.SPLITS for k in strata_split[s]})
    report = {
        "input": args.inp,
        "method": ("групповой стратифицированный сплит: рандомизированный "
                   "жадный поиск (GRASP), минимизация суммарной попарной JSD "
                   "распределений страт (lang × is_scam × onset_bucket)"),
        "master_seed": master_seed,
        "n_trials": n_trials,
        "n_valid_trials": result["n_valid_trials"],
        "best_trial": result["best_trial"],
        "fractions": fractions,
        "large_group_min_size": large_min,
        "large_groups_placement": result["large"],
        "jsd_bits": {
            "total": result["total_jsd"],
            "train_vs_val": sp.jsd(dists["train"], dists["val"]),
            "train_vs_test": sp.jsd(dists["train"], dists["test"]),
            "val_vs_test": sp.jsd(dists["val"], dists["test"]),
            "per_stratum_train_vs_test": {
                sp.stratum_label(k): v for k, v in
                sp.jsd_contributions(dists["train"], dists["test"], keys).items()},
            "per_stratum_train_vs_val": {
                sp.stratum_label(k): v for k, v in
                sp.jsd_contributions(dists["train"], dists["val"], keys).items()},
        },
        "checks": checks,
        "composition": composition_tables(records),
    }
    pathlib.Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.report).write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"отчёт: {args.report}")

    if not args.no_figures:
        try:
            make_figures(records, result, checks, args.figdir)
        except ImportError:
            print("matplotlib недоступен — фигуры пропущены "
                  "(pip install -e '.[viz]')")

    failed = [n for n, c in checks.items() if not c["pass"]]
    if failed:
        sys.exit(f"❌ сплит НЕ валиден, провалены: {failed}")
    print("✅ все проверки пройдены")


if __name__ == "__main__":
    main()
