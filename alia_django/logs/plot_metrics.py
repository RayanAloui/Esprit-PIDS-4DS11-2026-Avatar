"""
ALIA Metrics Log Visualizer
Usage: python plot_metrics.py [path/to/metrics.log]
Defaults to ./metrics.log if no argument is given.
"""

import re
import sys
import os
from datetime import datetime
from collections import defaultdict

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import MaxNLocator
import numpy as np

# ── 1. PARSE ──────────────────────────────────────────────────────────────────

LOG_RE = re.compile(
    r"(?P<level>\w+)\s+"
    r"(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d+)\s+"
    r"\S+\s+\[Metrics\]\s+"
    r"(?P<method>GET|POST|PUT|PATCH|DELETE)\s+"
    r"(?P<path>\S+)\s+"
    r"status=(?P<status>\d+)\s+"
    r"latency_s=(?P<latency>[\d.]+)\s+"
    r"cpu=(?P<cpu>[\d.]+)%\s+"
    r"rss_mb=(?P<rss_mb>[\d.]+)\s+"
    r"mem_pct=(?P<mem_pct>[\d.]+)\s+"
    r"sys_mem_pct=(?P<sys_mem_pct>[\d.]+)\s+"
    r"gpu=(?P<gpu>[\d.]+)%\s+"
    r"gpu_mem=(?P<gpu_mem>[\d.]+)%"
)


def parse_log(path: str) -> pd.DataFrame:
    records = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            m = LOG_RE.search(line)
            if m:
                d = m.groupdict()
                d["ts"] = datetime.strptime(d["ts"], "%Y-%m-%d %H:%M:%S,%f")
                for col in ("latency", "cpu", "rss_mb", "mem_pct", "sys_mem_pct", "gpu", "gpu_mem"):
                    d[col] = float(d[col])
                d["status"] = int(d["status"])
                records.append(d)
    if not records:
        raise ValueError("No matching log lines found – check the file path / format.")
    df = pd.DataFrame(records).sort_values("ts").reset_index(drop=True)
    return df


def classify_endpoint(path: str) -> str:
    """Group raw URL paths into readable endpoint labels."""
    if re.search(r"/static/audio/", path):
        return "Static Audio"
    mapping = {
        "/alia-api/ask_alia": "ask_alia",
        "/alia-api/listen":   "listen",
        "/alia-api/reset":    "reset",
        "/alia-api/":         "alia-api (root)",
        "/simulator/qcm/":    "QCM Questions",
        "/simulator/":        "Simulator",
        "/crm/":              "CRM",
        "/routes/":           "Routes",
        "/analytics/":        "Analytics",
        "/":                  "Home",
    }
    for prefix, label in mapping.items():
        if path.startswith(prefix):
            return label
    return path  # fallback: show full path


# ── 2. STYLE ──────────────────────────────────────────────────────────────────

DARK_BG   = "#0f1117"
PANEL_BG  = "#1a1d2e"
GRID_CLR  = "#2a2d3e"
TEXT_CLR  = "#e0e0f0"
ACCENT    = "#7c6af7"       # purple
GREEN     = "#3ecf8e"
YELLOW    = "#f5a623"
RED       = "#f25c5c"
CYAN      = "#4fc3f7"
ORANGE    = "#ff8a65"

ENDPOINT_PALETTE = [
    "#7c6af7", "#3ecf8e", "#f5a623", "#f25c5c",
    "#4fc3f7", "#ff8a65", "#a78bfa", "#34d399",
    "#fbbf24", "#f87171",
]

plt.rcParams.update({
    "figure.facecolor":  DARK_BG,
    "axes.facecolor":    PANEL_BG,
    "axes.edgecolor":    GRID_CLR,
    "axes.labelcolor":   TEXT_CLR,
    "axes.titlecolor":   TEXT_CLR,
    "axes.titlesize":    11,
    "axes.labelsize":    9,
    "axes.grid":         True,
    "grid.color":        GRID_CLR,
    "grid.linestyle":    "--",
    "grid.alpha":        0.6,
    "xtick.color":       TEXT_CLR,
    "ytick.color":       TEXT_CLR,
    "xtick.labelsize":   8,
    "ytick.labelsize":   8,
    "legend.facecolor":  PANEL_BG,
    "legend.edgecolor":  GRID_CLR,
    "legend.labelcolor": TEXT_CLR,
    "legend.fontsize":   8,
    "text.color":        TEXT_CLR,
    "font.family":       "monospace",
})


def _fmt_xaxis(ax, df):
    """Auto-format the time axis based on the span of the data."""
    span = (df["ts"].max() - df["ts"].min()).total_seconds()
    if span < 600:          # < 10 min  → show seconds
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    elif span < 3600:       # < 1 h     → show minutes
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    else:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d %H:%M"))
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")


# ── 3. INDIVIDUAL PLOTS ───────────────────────────────────────────────────────

def plot_avg_latency(ax, df):
    stats = (df.groupby("endpoint")["latency"]
               .agg(avg="mean", p95=lambda x: x.quantile(0.95))
               .sort_values("avg", ascending=True))
    colors = [ENDPOINT_PALETTE[i % len(ENDPOINT_PALETTE)]
              for i in range(len(stats))]
    bars = ax.barh(stats.index, stats["avg"], color=colors, alpha=0.85, zorder=3)
    # p95 error whiskers (right side only)
    for i, (ep, row) in enumerate(stats.iterrows()):
        ax.plot([row["avg"], row["p95"]], [i, i],
                color=RED, linewidth=2, solid_capstyle="round", zorder=4)
        ax.plot(row["p95"], i, marker="|",
                color=RED, markersize=10, markeredgewidth=2, zorder=5)
    # value labels
    for bar, val in zip(bars, stats["avg"]):
        ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}s", va="center", fontsize=8, color=TEXT_CLR)
    ax.set_title("Average Latency per Endpoint  (━ p95)")
    ax.set_xlabel("Latency (s)")
    ax.set_xlim(left=0)


def plot_memory_timeline(ax, df):
    ax.plot(df["ts"], df["rss_mb"],
            color=ACCENT, linewidth=1.8, label="RSS Memory (MB)")
    ax.fill_between(df["ts"], df["rss_mb"],
                    alpha=0.15, color=ACCENT)
    ax2 = ax.twinx()
    ax2.plot(df["ts"], df["sys_mem_pct"],
             color=CYAN, linewidth=1.4, linestyle="--",
             label="System Mem %")
    ax2.set_ylabel("System Memory %", color=CYAN)
    ax2.tick_params(axis="y", labelcolor=CYAN)
    ax2.spines["right"].set_edgecolor(GRID_CLR)
    ax.set_title("Memory Usage Over Time")
    ax.set_ylabel("RSS Memory (MB)")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
    _fmt_xaxis(ax, df)


def plot_gpu_timeline(ax, df):
    ax.plot(df["ts"], df["gpu"],
            color=GREEN, linewidth=1.8, label="GPU Utilization %")
    ax.fill_between(df["ts"], df["gpu"],
                    alpha=0.15, color=GREEN)
    ax.plot(df["ts"], df["gpu_mem"],
            color=YELLOW, linewidth=1.4, linestyle="--",
            label="GPU Memory %")
    ax.set_title("GPU Utilization & Memory Over Time")
    ax.set_ylabel("%")
    ax.legend(loc="upper left")
    _fmt_xaxis(ax, df)


def plot_latency_by_endpoint(ax, df):
    order = (df.groupby("endpoint")["latency"]
               .median()
               .sort_values(ascending=True)
               .index.tolist())
    data   = [df.loc[df["endpoint"] == ep, "latency"].values for ep in order]
    colors = [ENDPOINT_PALETTE[i % len(ENDPOINT_PALETTE)]
              for i in range(len(order))]
    bp = ax.boxplot(data, vert=False, patch_artist=True,
                    medianprops=dict(color=TEXT_CLR, linewidth=1.5),
                    whiskerprops=dict(color=GRID_CLR),
                    capprops=dict(color=GRID_CLR),
                    flierprops=dict(marker="x", color=RED, markersize=5))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    ax.set_yticklabels(order)
    ax.set_title("Latency Distribution per Endpoint")
    ax.set_xlabel("Latency (s)")


def plot_request_volume(ax, df):
    counts = df["endpoint"].value_counts()
    colors = [ENDPOINT_PALETTE[i % len(ENDPOINT_PALETTE)]
              for i in range(len(counts))]
    bars = ax.barh(counts.index[::-1], counts.values[::-1],
                   color=colors[::-1], alpha=0.85)
    for bar, val in zip(bars, counts.values[::-1]):
        ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height() / 2,
                str(val), va="center", fontsize=8, color=TEXT_CLR)
    ax.set_title("Request Volume by Endpoint")
    ax.set_xlabel("Number of Requests")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))


def plot_status_distribution(ax, df):
    counts = df["status"].value_counts()
    color_map = {200: GREEN, 201: CYAN, 400: YELLOW,
                 404: ORANGE, 500: RED, 503: RED}
    colors = [color_map.get(s, ACCENT) for s in counts.index]
    wedges, texts, autotexts = ax.pie(
        counts.values, labels=[str(s) for s in counts.index],
        colors=colors, autopct="%1.0f%%",
        startangle=90, pctdistance=0.75,
        wedgeprops=dict(edgecolor=DARK_BG, linewidth=2)
    )
    for t in texts + autotexts:
        t.set_color(TEXT_CLR)
        t.set_fontsize(9)
    ax.set_title("HTTP Status Distribution")


def plot_resource_heatmap(ax, df):
    """Per-endpoint average resource usage heatmap."""
    cols = ["latency", "rss_mb", "mem_pct", "sys_mem_pct", "gpu", "gpu_mem"]
    labels = ["Latency (s)", "RSS MB", "Mem %", "SysMem %", "GPU %", "GPUMem %"]
    pivot = df.groupby("endpoint")[cols].mean()
    # Normalize each column 0-1 for colour, but display raw values
    norm = (pivot - pivot.min()) / (pivot.max() - pivot.min() + 1e-9)
    im = ax.imshow(norm.values.T, aspect="auto",
                   cmap="plasma", vmin=0, vmax=1)
    ax.set_xticks(range(len(pivot.index)))
    ax.set_xticklabels(pivot.index, rotation=35, ha="right", fontsize=7.5)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    # Annotate with raw values
    for i, ep in enumerate(pivot.index):
        for j, col in enumerate(cols):
            val = pivot.loc[ep, col]
            fmt = f"{val:.1f}"
            ax.text(i, j, fmt, ha="center", va="center",
                    fontsize=7, color="white",
                    fontweight="bold" if norm.values[i, j] > 0.6 else "normal")
    ax.set_title("Avg Resource Usage per Endpoint (color = normalised)")
    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)


# ── 4. SAVE INDIVIDUAL PLOTS ──────────────────────────────────────────────────

def _save(plot_fn, df, out_path, figsize=(12, 5)):
    fig, ax = plt.subplots(figsize=figsize, facecolor=DARK_BG)
    ax.set_facecolor(PANEL_BG)
    plot_fn(ax, df)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"[✓] {os.path.basename(out_path)}")


def save_all(df: pd.DataFrame, base: str):
    os.makedirs(base, exist_ok=True)

    _save(plot_avg_latency,       df, f"{base}/01_avg_latency.png",      figsize=(12, 5))
    _save(plot_latency_by_endpoint, df, f"{base}/02_latency_distribution.png", figsize=(12, 5))
    _save(plot_memory_timeline,   df, f"{base}/03_memory_over_time.png", figsize=(13, 4))
    _save(plot_gpu_timeline,      df, f"{base}/04_gpu_over_time.png",    figsize=(13, 4))
    _save(plot_status_distribution, df, f"{base}/05_http_status.png",   figsize=(7,  7))
    _save(plot_request_volume,    df, f"{base}/06_request_volume.png",   figsize=(12, 5))

    # heatmap needs its own subplot size
    fig, ax = plt.subplots(figsize=(14, 5), facecolor=DARK_BG)
    ax.set_facecolor(PANEL_BG)
    plot_resource_heatmap(ax, df)
    fig.savefig(f"{base}/07_resource_heatmap.png", dpi=150,
                bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"[✓] 07_resource_heatmap.png")


# ── 5. SUMMARY STATS ──────────────────────────────────────────────────────────

def print_summary(df: pd.DataFrame):
    print("\n" + "═" * 56)
    print("  ALIA Metrics Summary")
    print("═" * 56)
    print(f"  Period  : {df['ts'].min()}  →  {df['ts'].max()}")
    print(f"  Requests: {len(df)}")
    print(f"  Endpoints: {df['endpoint'].nunique()}")
    print(f"  Status 2xx: {(df['status'] // 100 == 2).sum()} / {len(df)}")
    print()
    print(f"  Latency  avg={df['latency'].mean():.3f}s  "
          f"p50={df['latency'].median():.3f}s  "
          f"p95={df['latency'].quantile(.95):.3f}s  "
          f"max={df['latency'].max():.3f}s")
    print(f"  RSS MB   avg={df['rss_mb'].mean():.1f}  "
          f"max={df['rss_mb'].max():.1f}")
    print(f"  GPU util avg={df['gpu'].mean():.1f}%  "
          f"max={df['gpu'].max():.1f}%")
    print("═" * 56 + "\n")


# ── 6. MAIN ───────────────────────────────────────────────────────────────────

def main():
    log_path = sys.argv[1] if len(sys.argv) > 1 else "metrics.log"

    if not os.path.isfile(log_path):
        print(f"[✗] File not found: {log_path}")
        sys.exit(1)

    print(f"[…] Parsing {log_path} …")
    df = parse_log(log_path)
    df["endpoint"] = df["path"].apply(classify_endpoint)

    print_summary(df)

    base = os.path.splitext(log_path)[0] + "_plots"
    print(f"\n[…] Saving plots to ./{base}/\n")
    save_all(df, base)
    print(f"\nDone! {base}/ contains 7 PNG files.\n")


if __name__ == "__main__":
    main()
