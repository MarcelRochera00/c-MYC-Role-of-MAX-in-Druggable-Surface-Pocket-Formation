#!/usr/bin/env python3
"""
individual_pocket_analysis.py

Batch analysis pipeline for mdpocket's per-frame descriptor output, across
every sub-pocket produced by an mdpocket run (SP1, SP2, ... SPn).

For each pocket folder found under ROOT_DIR (e.g. Results/Pockets/SP1,
Results/Pockets/SP2, ...), this script:
  1. Finds and parses that pocket's descriptor file, named
     "<pocket_name>_descriptors.txt" (e.g. SP1_descriptors.txt inside the
     SP1 folder) - not a bare "_descriptors.txt", since mdpocket writes
     every per-pocket file with the pocket name as a prefix and several
     of these prefixed files (descriptors, atoms, info, ...) commonly sit
     side by side in the same folder.
     (parsing preserves the original manual, nothing-silently-dropped
     logic: malformed lines are reported and skipped, short "pocket not
     detected" rows are kept with pock_volume forced to 0).
  2. Adds a time_ns column (time_ns = snapshot * TIME_PER_SNAPSHOT_NS).
  3. Writes that pocket's clean descriptors.csv, summary_statistics.csv,
     and a 4-panel time-series figure into its own output subfolder.

Across all pockets, it then:
  4. Writes one combined pocket_summary_statistics.csv (long format: one
     row per pocket x descriptor, with mean/median/std/min/max/CV).
  5. Writes one combined 2x3-grid figure per descriptor (SP1 ... SPn),
     with identical x/y limits across panels so pockets are visually
     comparable, at publication-quality (600 DPI).
  6. Writes one PNG "quick-look" table (4.6_pocket_means_table.png) with
     the mean of the 4 plotted descriptors for every pocket, formatted as
     a readable table image (colored header, striped rows) for fast
     visual reference alongside the CSVs.

Just set ROOT_DIR (and TIME_PER_SNAPSHOT_NS, if different) below and run:
    python individual_pocket_analysis.py

"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import NamedTuple

import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# >>> CONFIG - the only section you should need to touch <<<
# ============================================================

# Folder that directly contains the SP1, SP2, ... SPn pocket subfolders.
ROOT_DIR = Path("Results/Pockets")

# Glob pattern used to discover pocket subfolders under ROOT_DIR.
POCKET_DIR_PATTERN = "SP*"

# Suffix of the descriptor filename (as written by mdpocket) inside each
# pocket folder. The actual file mdpocket writes is prefixed with the
# pocket's own name, e.g. "SP1_descriptors.txt" inside the SP1 folder,
# "SP2_descriptors.txt" inside SP2, etc. - so the filename actually
# looked up is "<pocket_name>" + DESCRIPTOR_SUFFIX, built per pocket in
# find_descriptor_file() below. A bare "_descriptors.txt" (no prefix)
# would also match other pockets' prefixed files or unrelated
# "*_descriptors.txt" files if several sub-pockets' outputs happen to
# live in the same folder, so the prefix match is required, not optional.
DESCRIPTOR_SUFFIX = "_descriptors.txt"

# Name of the folder (created under ROOT_DIR) that all outputs go into.
OUTPUT_DIRNAME = "sub_pockets_analysis"

# Trajectory timing: time_ns = snapshot_index * TIME_PER_SNAPSHOT_NS.
# Example: 500 ns trajectory saved every 50 ps -> 0.05 ns/snapshot.
TIME_PER_SNAPSHOT_NS = 0.05

# Rolling-mean smoothing window, in nanoseconds (converted to a frame
# count using TIME_PER_SNAPSHOT_NS). Kept within the requested 5-10 ns range.
ROLLING_WINDOW_NS = 7.5

# If True, frames where the pocket was NOT detected (pock_volume == 0) are
# excluded from summary statistics and from rolling means / axis limits.
# The raw per-frame CSV always keeps every successfully parsed row (zeros
# included) regardless of this flag.
EXCLUDE_UNDETECTED_FRAMES_FROM_STATS = False

# Figure DPI for saved plots.
FIGURE_DPI = 600

# ============================================================
# End of CONFIG
# ============================================================

# Canonical column names written by mdpocket (see mdpocket.h, M_MDP_OUTP_HEADER)
EXPECTED_COLUMNS = [
    "snapshot", "pock_volume", "pock_asa", "pock_pol_asa", "pock_apol_asa",
    "pock_asa22", "pock_pol_asa22", "pock_apol_asa22", "nb_AS", "mean_as_ray",
    "mean_as_solv_acc", "apol_as_prop", "mean_loc_hyd_dens",
    "hydrophobicity_score", "volume_score", "polarity_score", "charge_score",
    "prop_polar_atm", "as_density", "as_max_dst",
]


class DescriptorPlot(NamedTuple):
    column: str
    title: str
    ylabel: str
    color: str
    filename_stub: str


# The 4 descriptors plotted, in order, with their titles / y-axis labels /
# colors, and the stub used to name the combined per-descriptor figure.
PLOTS = [
    DescriptorPlot("pock_volume", "Pocket Volume", "Volume (\u00c5\u00b3)",
                    "#2E86AB", "4.1_combined_pocket_volume"),
    DescriptorPlot("hydrophobicity_score", "Hydrophobicity Score", "Hydrophobicity score",
                    "#A23B72", "4.2_combined_hydrophobicity_score"),
    DescriptorPlot("mean_loc_hyd_dens", "Mean Local Hydrophobic Density", "Density",
                    "#F18F01", "4.3_combined_mean_loc_hyd_dens"),
    DescriptorPlot("polarity_score", "Polarity Score", "Polarity score",
                    "#3B7A57", "4.4_combined_polarity_score"),
]


class PocketResult(NamedTuple):
    """Everything produced for one successfully-processed pocket."""
    name: str
    df: pd.DataFrame
    stats: pd.DataFrame


class Diagnostics:
    """Accumulates the parser/report diagnostics for one pocket, and can
    print a per-pocket report or fold into a run-wide summary."""

    def __init__(self, pocket_name: str):
        self.pocket_name = pocket_name
        self.total_lines = 0
        self.n_parsed_rows = 0
        self.n_short_rows = 0
        self.malformed: list[tuple[int, int, str]] = []
        self.n_zero_volume = 0
        self.n_total_rows = 0
        self.mean_volume_incl = float("nan")
        self.mean_volume_excl = float("nan")

    @property
    def pct_undetected(self) -> float:
        if self.n_total_rows == 0:
            return float("nan")
        return 100 * self.n_zero_volume / self.n_total_rows

    def print_report(self) -> None:
        print(f"[info] Total lines in file (incl. header): {self.total_lines}")
        print(f"[info] Successfully parsed data rows: {self.n_parsed_rows} "
              f"({self.n_short_rows} of these were short 'pocket not detected' rows, "
              f"kept as pock_volume=0)")
        if self.malformed:
            print(f"[warning] {len(self.malformed)} line(s) could not be parsed and were "
                  f"SKIPPED (not silently averaged in, not silently coerced):")
            for lineno, nfields, content in self.malformed[:10]:
                print(f"    line {lineno}: expected fields mismatch, got {nfields} -> {content!r}")
            if len(self.malformed) > 10:
                print(f"    ... and {len(self.malformed) - 10} more.")
        if self.n_total_rows:
            print(f"[info] Frames with pock_volume == 0 (pocket not detected): "
                  f"{self.n_zero_volume} / {self.n_total_rows} ({self.pct_undetected:.1f}%)")
            print(f"[info] Mean pock_volume INCLUDING zero/undetected frames: {self.mean_volume_incl:.2f}")
            print(f"[info] Mean pock_volume EXCLUDING zero/undetected frames: {self.mean_volume_excl:.2f}")


def natural_sp_sort_key(path: Path):
    """Sort pocket folders numerically by the digits in their name (SP2
    before SP10), falling back to plain alphabetical for anything that
    doesn't end in digits."""
    match = re.search(r"(\d+)\s*$", path.name)
    if match:
        return (0, int(match.group(1)), path.name)
    return (1, 0, path.name)


def discover_pocket_dirs(root: Path, pattern: str) -> list[Path]:
    """Every subfolder of `root` matching `pattern`, sorted naturally
    (SP1, SP2, ... SP10, ...)."""
    if not root.exists():
        raise FileNotFoundError(f"ROOT_DIR does not exist: {root}")
    dirs = [p for p in root.glob(pattern) if p.is_dir()]
    return sorted(dirs, key=natural_sp_sort_key)


def find_descriptor_file(folder: Path, pocket_name: str, suffix: str) -> Path | None:
    """Locate the descriptor file for one pocket folder. mdpocket names
    this file "<pocket_name><suffix>" (e.g. "SP1_descriptors.txt" inside
    the SP1 folder) - other prefixed files for the same or other pockets
    (e.g. SP1_atoms.txt, SP1_info.txt, or another pocket's own
    "SPn_descriptors.txt") can live in the same folder, so we look up the
    exact prefixed name rather than any file merely ending in `suffix`.
    Returns None (rather than raising) so the caller can report it as a
    missing pocket and continue with the rest of the batch."""
    expected_name = f"{pocket_name}{suffix}"
    candidate = folder / expected_name
    if candidate.exists():
        return candidate
    matches = list(folder.rglob(expected_name))
    return matches[0] if matches else None


def load_descriptors(filepath: Path, pocket_name: str) -> tuple[pd.DataFrame, Diagnostics]:
    """
    Manually parse an mdpocket descriptors file line by line. Nothing is
    silently dropped: every non-blank line after the header is either
    parsed into a row, or recorded as malformed in the returned
    Diagnostics. Short rows (pocket not detected that frame) are kept,
    with pock_volume forced to 0.0 and every other column set to NaN.
    """
    diag = Diagnostics(pocket_name)

    with open(filepath, "r") as fh:
        raw_lines = [ln.rstrip("\n") for ln in fh if ln.strip()]

    if not raw_lines:
        raise ValueError(f"{filepath} is empty.")
    diag.total_lines = len(raw_lines)

    # header: strip a leading '#' token if present (mdpocket writes
    # '# snapshot pock_volume ...' with '#' as its own token)
    header_tokens = raw_lines[0].split()
    if header_tokens[0] == "#":
        header_tokens = header_tokens[1:]
    header = header_tokens
    n_cols = len(header)

    if header != EXPECTED_COLUMNS:
        print(f"[warning] [{pocket_name}] Header does not exactly match the expected mdpocket "
              f"column order. Proceeding with the header as found in the file.")

    rows = []
    for i, line in enumerate(raw_lines[1:], start=2):  # start=2 -> real file line number
        fields = line.split()

        if len(fields) == n_cols:
            try:
                row = [float(f) for f in fields]
            except ValueError:
                diag.malformed.append((i, len(fields), line))
                continue
            rows.append(row)

        elif len(fields) < n_cols:
            # short line for a frame where the reference pocket was NOT
            # detected (typically just 'snapshot 0.00' or similar) -
            # recognized rather than silently discarded, since discarding
            # would exclude real pocket-closure events and inflate the mean.
            try:
                partial = [float(f) for f in fields]
            except ValueError:
                diag.malformed.append((i, len(fields), line))
                continue
            row = partial + [float("nan")] * (n_cols - len(partial))
            if len(partial) < 2:
                row[1] = 0.0
            rows.append(row)
            diag.n_short_rows += 1

        else:
            # more fields than expected - genuinely malformed, don't guess
            diag.malformed.append((i, len(fields), line))

    diag.n_parsed_rows = len(rows)

    df = pd.DataFrame(rows, columns=header)
    if "snapshot" in df.columns:
        df = df.sort_values("snapshot").reset_index(drop=True)

    if "pock_volume" in df.columns:
        diag.n_total_rows = len(df)
        diag.n_zero_volume = int((df["pock_volume"] == 0).sum())
        diag.mean_volume_incl = df["pock_volume"].mean()
        if diag.n_zero_volume < diag.n_total_rows:
            diag.mean_volume_excl = df.loc[df["pock_volume"] > 0, "pock_volume"].mean()

    return df, diag


def add_time_column(df: pd.DataFrame, time_per_snapshot_ns: float) -> pd.DataFrame:
    """Insert a time_ns column right after snapshot: time_ns = snapshot * time_per_snapshot_ns."""
    df = df.copy()
    df.insert(1, "time_ns", df["snapshot"] * time_per_snapshot_ns)
    return df


def summarize(df: pd.DataFrame, exclude_undetected: bool) -> pd.DataFrame:
    """Per-descriptor mean/std/min/max/median for one pocket."""
    data = df
    if exclude_undetected and "pock_volume" in df.columns:
        data = df.loc[df["pock_volume"] > 0]
    numeric_cols = [c for c in data.columns if c not in ("snapshot", "time_ns")]
    stats = data[numeric_cols].agg(["mean", "std", "min", "max", "median"]).T
    stats.index.name = "descriptor"
    return stats.round(4)


def rolling_window_frames(rolling_window_ns: float, time_per_snapshot_ns: float) -> int:
    return max(1, round(rolling_window_ns / time_per_snapshot_ns))


def style_axis(ax: plt.Axes) -> None:
    """Shared publication-style cleanup applied to every axis."""
    ax.grid(alpha=0.3, linestyle=":")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def draw_trajectory(ax: plt.Axes, time_ns, y, color: str, window_frames: int) -> None:
    """Thin grey raw trajectory behind a thicker colored rolling mean -
    shared by both the per-pocket panels and the combined grid figures."""
    ax.plot(time_ns, y, linewidth=0.6, color="0.7", alpha=0.8, zorder=1, label="raw")
    rolled = pd.Series(y).rolling(window_frames, center=True, min_periods=1).mean()
    ax.plot(time_ns, rolled, linewidth=2.2, color=color, zorder=3,
             solid_capstyle="round", label=f"rolling mean ({window_frames} frames)")


def plot_single_pocket(df: pd.DataFrame, pocket_name: str, outpath: Path,
                        exclude_undetected: bool, window_frames: int) -> None:
    """4-panel time-series figure (one panel per descriptor) for a single pocket."""
    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "axes.titleweight": "bold",
    })

    plot_df = df
    if exclude_undetected and "pock_volume" in df.columns:
        plot_df = df.loc[df["pock_volume"] > 0]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes = axes.flatten()

    for ax, spec in zip(axes, PLOTS):
        if spec.column not in plot_df.columns or plot_df[spec.column].isna().all():
            ax.set_title(f"{spec.title}\n(data not found)", color="gray")
            ax.axis("off")
            continue

        time_ns = plot_df["time_ns"].to_numpy()
        y = plot_df[spec.column].to_numpy()
        draw_trajectory(ax, time_ns, y, spec.color, window_frames)

        mean_val = float(pd.Series(y).mean())
        ax.axhline(mean_val, linestyle="--", linewidth=1.3, color="black",
                    alpha=0.6, zorder=2, label=f"mean = {mean_val:.2f}")

        ax.set_title(f"{spec.title} Over Time")
        ax.set_xlabel("Time (ns)")
        ax.set_ylabel(spec.ylabel)
        ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
        style_axis(ax)

    suffix = " (excl. undetected frames)" if exclude_undetected else ""
    fig.suptitle(f"{pocket_name} - MDpocket Descriptors Over the Trajectory{suffix}",
                 fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(outpath, dpi=FIGURE_DPI)
    plt.close(fig)


def grid_shape(n: int) -> tuple[int, int]:
    """(nrows, ncols) for the combined grid: up to 3 columns per row, which
    gives exactly a 2x3 grid for the 6-pocket case this was built for, and
    degrades sensibly for other pocket counts."""
    ncols = min(3, n) if n > 0 else 1
    nrows = math.ceil(n / ncols) if ncols else 1
    return nrows, ncols


def plot_combined_descriptor(results: list[PocketResult], spec: DescriptorPlot,
                              outpath: Path, exclude_undetected: bool,
                              window_frames: int) -> None:
    """One figure per descriptor, one panel per pocket (SP1 ... SPn), with
    identical x and y limits across all panels so pockets are directly
    comparable."""
    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "axes.titleweight": "bold",
    })

    # gather per-pocket (time, y) series, applying the undetected-frame
    # filter consistently with the rest of the script
    series_by_pocket = {}
    for res in results:
        data = res.df
        if exclude_undetected and "pock_volume" in data.columns:
            data = data.loc[data["pock_volume"] > 0]
        if spec.column in data.columns:
            series_by_pocket[res.name] = (data["time_ns"].to_numpy(), data[spec.column].to_numpy())

    if not series_by_pocket:
        print(f"[warning] Descriptor '{spec.column}' not found in any pocket - skipping "
              f"combined figure {outpath.name}.")
        return

    # shared axis limits across every panel in this figure
    all_x = pd.concat([pd.Series(t) for t, _ in series_by_pocket.values()])
    all_y = pd.concat([pd.Series(y) for _, y in series_by_pocket.values()])
    x_min, x_max = float(all_x.min()), float(all_x.max())
    y_min, y_max = float(all_y.min(skipna=True)), float(all_y.max(skipna=True))
    y_pad = 0.05 * (y_max - y_min) if y_max > y_min else 1.0

    nrows, ncols = grid_shape(len(results))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3.6 * nrows),
                              sharex=True, sharey=True, squeeze=False)
    axes = axes.flatten()

    for ax, res in zip(axes, results):
        if res.name not in series_by_pocket:
            ax.set_title(f"{res.name}\n(data not found)", color="gray")
            ax.axis("off")
            continue

        time_ns, y = series_by_pocket[res.name]
        draw_trajectory(ax, time_ns, y, spec.color, window_frames)
        mean_val = float(pd.Series(y).mean())
        ax.axhline(mean_val, linestyle="--", linewidth=1.1, color="black", alpha=0.6, zorder=2)

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min - y_pad, y_max + y_pad)
        ax.set_title(res.name)
        style_axis(ax)

    # hide any unused trailing axes if pocket count doesn't fill the grid
    for ax in axes[len(results):]:
        ax.axis("off")

    for ax in axes[: len(results)][-ncols:]:
        ax.set_xlabel("Time (ns)")
    for row_start in range(0, len(axes), ncols):
        axes[row_start].set_ylabel(spec.ylabel)

    suffix = " (excl. undetected frames)" if exclude_undetected else ""
    fig.suptitle(f"{spec.title} Across Sub-Pockets{suffix}", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(outpath, dpi=FIGURE_DPI)
    plt.close(fig)


def plot_all_descriptors_grid(results: list[PocketResult], specs: list[DescriptorPlot],
                               outpath: Path, exclude_undetected: bool,
                               window_frames: int) -> None:
    """One single figure: rows = the 4 descriptors, columns = every pocket
    (SP1 ... SPn). Y-limits are shared across a row (so pockets are
    comparable per descriptor); x-limits are shared across the whole
    figure (same trajectory)."""
    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 9,
        "axes.titleweight": "bold",
    })

    # filtered (time, per-descriptor-values) per pocket, applying the same
    # undetected-frame filter used everywhere else
    filtered = {}
    for res in results:
        data = res.df
        if exclude_undetected and "pock_volume" in data.columns:
            data = data.loc[data["pock_volume"] > 0]
        filtered[res.name] = data

    # x-limits shared across the entire figure
    all_x = pd.concat([data["time_ns"] for data in filtered.values()])
    x_min, x_max = float(all_x.min()), float(all_x.max())

    nrows, ncols = len(specs), len(results)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.4 * ncols, 2.8 * nrows),
                              sharex=True, squeeze=False)

    for i, spec in enumerate(specs):
        # y-limits shared within this descriptor's row only
        row_series = {name: data[spec.column] for name, data in filtered.items()
                       if spec.column in data.columns}
        if row_series:
            all_y = pd.concat(row_series.values())
            y_min, y_max = float(all_y.min(skipna=True)), float(all_y.max(skipna=True))
            y_pad = 0.05 * (y_max - y_min) if y_max > y_min else 1.0
        else:
            y_min = y_max = y_pad = 0.0

        for j, res in enumerate(results):
            ax = axes[i, j]
            data = filtered.get(res.name)
            if data is None or spec.column not in data.columns or data[spec.column].isna().all():
                ax.set_title("(no data)" if i == 0 else "", color="gray", fontsize=9)
                ax.axis("off")
                continue

            time_ns = data["time_ns"].to_numpy()
            y = data[spec.column].to_numpy()
            draw_trajectory(ax, time_ns, y, spec.color, window_frames)
            mean_val = float(pd.Series(y).mean())
            ax.axhline(mean_val, linestyle="--", linewidth=1.0, color="black", alpha=0.6, zorder=2)

            ax.set_xlim(x_min, x_max)
            if row_series:
                ax.set_ylim(y_min - y_pad, y_max + y_pad)
            style_axis(ax)

            if i == 0:
                ax.set_title(res.name)
            if j == 0:
                ax.set_ylabel(spec.ylabel)
            if i == nrows - 1:
                ax.set_xlabel("Time (ns)")

    suffix = " (excl. undetected frames)" if exclude_undetected else ""
    fig.suptitle(f"All Descriptors Across All Sub-Pockets{suffix}", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(outpath, dpi=FIGURE_DPI)
    plt.close(fig)


def plot_pocket_means_table(results: list[PocketResult], specs: list[DescriptorPlot],
                             outpath: Path, exclude_undetected: bool) -> None:
    """Quick-look PNG table: one row per pocket, one column per plotted
    descriptor, cell value = that pocket's mean (matches the 'Mean' column
    of pocket_summary_statistics.csv / pocket_means_summary.txt, just laid
    out for fast visual scanning instead of parsing a CSV or text file).
    """
    plt.rcParams.update({"font.size": 11})

    def col_label(spec: DescriptorPlot) -> str:
        # Only show a unit line under the title when the ylabel actually
        # contains a real unit (e.g. "(Å³)" for volume) - not a plain word
        # like "(Density)", which is just the descriptor name again.
        unit_match = re.search(r"\(([^)]+)\)", spec.ylabel)
        if unit_match and re.search(r"[^A-Za-z\s]", unit_match.group(1)):
            return f"{spec.title}\n({unit_match.group(1)})"
        return spec.title

    col_labels = [col_label(spec) for spec in specs]
    row_labels = [res.name for res in results]

    cell_text = []
    for res in results:
        row = []
        for spec in specs:
            if spec.column in res.stats.index:
                mean_val = res.stats.loc[spec.column, "mean"]
                row.append(f"{mean_val:.2f}" if pd.notna(mean_val) else "n/a")
            else:
                row.append("n/a")
        cell_text.append(row)

    n_rows, n_cols = len(results), len(specs)
    fig_width = 2.6 * n_cols + 2.0
    fig_height = 0.9 + 0.6 * n_rows
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis("off")

    table = ax.table(
        cellText=cell_text,
        rowLabels=row_labels,
        colLabels=col_labels,
        cellLoc="center",
        rowLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.0)

    header_color = "#2E4057"
    stripe_color = "#EEF3F7"

    # header row (col labels live in row index 0)
    for j in range(n_cols):
        cell = table[0, j]
        cell.set_facecolor(header_color)
        cell.set_text_props(color="white", weight="bold")
        cell.set_edgecolor("white")

    # row-label column (matplotlib places these at column index -1)
    for i in range(n_rows):
        cell = table[i + 1, -1]
        cell.set_facecolor(header_color)
        cell.set_text_props(color="white", weight="bold")
        cell.set_edgecolor("white")

    # alternating stripes + column accent colors on the data cells
    for i in range(n_rows):
        for j in range(n_cols):
            cell = table[i + 1, j]
            cell.set_edgecolor("white")
            cell.set_facecolor(stripe_color if i % 2 == 0 else "white")
            cell.get_text().set_color(specs[j].color)
            cell.get_text().set_weight("bold")

    fig.tight_layout()
    fig.savefig(outpath, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)


def write_pocket_means_summary(results: list[PocketResult], outpath: Path) -> None:
    """Plain-text summary of each pocket's descriptor means - the quick
    'what are the headline numbers' reference to go with the CSVs."""
    lines = ["Pocket Descriptor Means - Summary", "=" * 70]
    for res in results:
        lines.append("")
        lines.append(res.name)
        lines.append("-" * len(res.name))
        for descriptor, row in res.stats.iterrows():
            mean_val = row["mean"]
            mean_str = f"{mean_val:.4f}" if pd.notna(mean_val) else "n/a"
            lines.append(f"  {descriptor:<25s}: {mean_str}")
    outpath.write_text("\n".join(lines) + "\n")


def compute_combined_summary(results: list[PocketResult]) -> pd.DataFrame:
    """Long-format table: one row per (pocket, descriptor), with mean,
    median, std, min, max, and coefficient of variation (std / mean)."""
    records = []
    for res in results:
        for descriptor, row in res.stats.iterrows():
            mean = row["mean"]
            std = row["std"]
            cv = (std / mean) if mean not in (0, float("nan")) and pd.notna(mean) else float("nan")
            records.append({
                "Pocket": res.name,
                "Descriptor": descriptor,
                "Mean": mean,
                "Median": row["median"],
                "Std": std,
                "Min": row["min"],
                "Max": row["max"],
                "Coefficient_of_Variation": cv,
            })
    return pd.DataFrame.from_records(records)


def process_pocket(pocket_dir: Path, output_root: Path) -> PocketResult | None:
    """Full single-pocket pipeline: locate + parse the descriptor file,
    add the time column, write descriptors.csv / summary_statistics.csv /
    the 4-panel figure, and return the result for use in the combined
    (cross-pocket) outputs. Returns None (after printing a warning) if the
    pocket's descriptor file can't be found."""
    pocket_name = pocket_dir.name
    print(f"\n{'=' * 70}\n[info] Processing {pocket_name}\n{'=' * 70}")

    descriptor_file = find_descriptor_file(pocket_dir, pocket_name, DESCRIPTOR_SUFFIX)
    if descriptor_file is None:
        print(f"[warning] No '{pocket_name}{DESCRIPTOR_SUFFIX}' found under {pocket_dir} - "
              f"skipping {pocket_name}.")
        return None

    print(f"[info] Reading: {descriptor_file}")
    df, diag = load_descriptors(descriptor_file, pocket_name)
    diag.print_report()

    df = add_time_column(df, TIME_PER_SNAPSHOT_NS)
    stats = summarize(df, EXCLUDE_UNDETECTED_FRAMES_FROM_STATS)

    pocket_outdir = output_root / pocket_name
    pocket_outdir.mkdir(parents=True, exist_ok=True)

    df.to_csv(pocket_outdir / "descriptors.csv", index=False)
    stats.to_csv(pocket_outdir / "summary_statistics.csv")

    window_frames = rolling_window_frames(ROLLING_WINDOW_NS, TIME_PER_SNAPSHOT_NS)
    plot_path = pocket_outdir / f"{pocket_name}_4panel.png"
    plot_single_pocket(df, pocket_name, plot_path, EXCLUDE_UNDETECTED_FRAMES_FROM_STATS, window_frames)

    print(f"[info] Wrote: {pocket_outdir / 'descriptors.csv'}")
    print(f"[info] Wrote: {pocket_outdir / 'summary_statistics.csv'}")
    print(f"[info] Wrote: {plot_path}")

    return PocketResult(name=pocket_name, df=df, stats=stats)


def main() -> None:
    output_root = ROOT_DIR / OUTPUT_DIRNAME
    output_root.mkdir(parents=True, exist_ok=True)

    pocket_dirs = discover_pocket_dirs(ROOT_DIR, POCKET_DIR_PATTERN)
    print(f"[info] Found {len(pocket_dirs)} pocket folder(s) under {ROOT_DIR} "
          f"matching '{POCKET_DIR_PATTERN}': {[p.name for p in pocket_dirs]}")

    results: list[PocketResult] = []
    missing_pockets: list[str] = []
    for pocket_dir in pocket_dirs:
        result = process_pocket(pocket_dir, output_root)
        if result is None:
            missing_pockets.append(pocket_dir.name)
        else:
            results.append(result)

    if not results:
        print("\n[error] No pockets were successfully processed - nothing to combine. "
              "Check ROOT_DIR and DESCRIPTOR_SUFFIX at the top of the script.")
        return

    print(f"\n{'=' * 70}\n[info] Building combined outputs across {len(results)} pocket(s)\n{'=' * 70}")

    combined_stats = compute_combined_summary(results)
    combined_csv = output_root / "pocket_summary_statistics.csv"
    combined_stats.to_csv(combined_csv, index=False)
    print(f"[info] Wrote: {combined_csv}")

    means_txt = output_root / "pocket_means_summary.txt"
    write_pocket_means_summary(results, means_txt)
    print(f"[info] Wrote: {means_txt}")

    window_frames = rolling_window_frames(ROLLING_WINDOW_NS, TIME_PER_SNAPSHOT_NS)
    for spec in PLOTS:
        outpath = output_root / f"{spec.filename_stub}.png"
        plot_combined_descriptor(results, spec, outpath, EXCLUDE_UNDETECTED_FRAMES_FROM_STATS,
                                  window_frames)
        print(f"[info] Wrote: {outpath}")

    grand_outpath = output_root / "4.5_combined_all_descriptors.png"
    plot_all_descriptors_grid(results, PLOTS, grand_outpath, EXCLUDE_UNDETECTED_FRAMES_FROM_STATS,
                               window_frames)
    print(f"[info] Wrote: {grand_outpath}")

    means_table_outpath = output_root / "4.6_pocket_means_table.png"
    plot_pocket_means_table(results, PLOTS, means_table_outpath, EXCLUDE_UNDETECTED_FRAMES_FROM_STATS)
    print(f"[info] Wrote: {means_table_outpath}")

    print(f"\n{'=' * 70}\n[done] Batch summary\n{'=' * 70}")
    print(f"  Pockets processed : {len(results)}  ({', '.join(r.name for r in results)})")
    if missing_pockets:
        print(f"  Pockets skipped (missing '<pocket>{DESCRIPTOR_SUFFIX}'): "
              f"{len(missing_pockets)}  ({', '.join(missing_pockets)})")
    else:
        print("  Pockets skipped   : none")
    print(f"  All outputs under : {output_root}")


if __name__ == "__main__":
    main()