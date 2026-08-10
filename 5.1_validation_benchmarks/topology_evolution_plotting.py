"""Shared plotting helpers for the KAN topology-evolution figure.

Used by both generate_topology_evolution_plot.sh (full HPC run) and
test_topology_evolution_plot.ipynb (local sanity-check run) so the two
never drift apart visually. Loaded via importlib from an explicit path in
the .sh script (see ROOT_DIR handling there) and via a normal import in
the notebook.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnnotationBbox, OffsetImage


def gray_fraction_series(design, low=0.05, high=0.95):
    return ((design > low) & (design < high)).mean(dim=("x", "y"))


def nearest_step(ds, target_step):
    return int(ds.step.sel(step=min(target_step, int(ds.step.values[-1])), method="nearest").values)


def add_topology_thumbnails(ax, ds, y_da, milestone_steps, thumb_width_in=0.28, y_pad_frac=0.65,
                             compliance_color="purple", gray_color="crimson"):
    """Draw small design-field thumbnails above the curve, with the exact
    (absolute) compliance and gray-fraction values for that step printed
    directly under each thumbnail as bare numbers -- color (purple vs.
    crimson, matching the two curves) is what identifies which value is
    which, not a text label.

    Thumbnails are placed at evenly spaced x-slots across the axis width
    rather than directly above their true step position, with an arrow
    pointing back to the actual data point. This keeps them from
    overlapping/stacking when milestone steps are close together (e.g. a
    short 50-step test with milestones at 10/20/30/50). The value labels are
    anchored to the same slot (not to the curve, where nearby milestones
    would stack their labels on top of each other once the curve flattens
    out near convergence).

    ``thumb_width_in`` is the on-page width of each thumbnail in inches,
    independent of the design's mesh resolution. OffsetImage's own ``zoom``
    parameter scales directly with the array's pixel count, so a raw zoom
    value tuned on a small test mesh (e.g. 60x20) would render many times
    larger -- and overlap -- on the real 256x192/256x256 problem meshes.
    Computing zoom from the actual array width keeps the physical thumbnail
    size (and therefore the "no overlap" guarantee) the same at any resolution.
    """
    ax.figure.canvas.draw()
    y_lo, y_hi = ax.get_ylim()
    y_span = y_hi - y_lo
    thumb_y = y_hi + y_span * (y_pad_frac * 0.5)
    ax.set_ylim(y_lo, y_hi + y_span * y_pad_frac)

    x_lo, x_hi = ax.get_xlim()
    x_span = x_hi - x_lo
    n = len(milestone_steps)
    slot_x = [x_lo + x_span * (0.12 + 0.76 * i / max(n - 1, 1)) for i in range(n)]

    for target_step, tx in zip(milestone_steps, slot_x):
        step = nearest_step(ds, target_step)
        design = ds.design.sel(step=step).values.T
        y_val = float(y_da.sel(step=step).values)
        compliance_val = float(ds.loss.sel(step=step).values)
        gray_val = float(ds.gray_fraction.sel(step=step).values)

        # zoom is in points-per-pixel (1 point = 1/72 inch); dividing the
        # target width by the array's pixel width makes the rendered size
        # resolution-independent.
        zoom = (thumb_width_in * 72.0) / design.shape[1]

        # cmap="Greys": low density (void) -> white, high density (material)
        # -> black, so the background reads white and the structure reads black.
        imagebox = OffsetImage(design, cmap="Greys", norm=plt.Normalize(vmin=0.0, vmax=1.0), zoom=zoom)
        imagebox.image.axes = ax
        ab = AnnotationBbox(
            imagebox,
            (step, y_val),
            xybox=(tx, thumb_y),
            xycoords="data",
            boxcoords="data",
            frameon=True,
            pad=0.2,
            bboxprops=dict(edgecolor="0.4", linewidth=0.8, facecolor="white"),
            arrowprops=dict(arrowstyle="->", color="0.35", lw=0.9, shrinkA=0, shrinkB=4),
            annotation_clip=False,
        )
        ax.add_artist(ab)
        ax.annotate(
            f"Step {step}",
            xy=(tx, thumb_y), xycoords="data",
            xytext=(0, 30), textcoords="offset points",
            ha="center", va="bottom", fontsize=8, color="0.25",
            annotation_clip=False,
        )

        # Value labels go directly under the thumbnail they describe (fixed
        # point offset below the box, scaled to the box's own rendered
        # height so it clears the frame at any mesh aspect ratio), instead
        # of on the curve where milestones close together would overlap.
        # Bare numbers only -- the color (matching each curve) is the label.
        box_half_height_pt = (design.shape[0] * zoom) / 2.0 + 4.0
        ax.annotate(
            f"{compliance_val:.4f}",
            xy=(tx, thumb_y), xycoords="data",
            xytext=(0, -(box_half_height_pt + 4)), textcoords="offset points",
            ha="center", va="top", fontsize=8, color=compliance_color,
            annotation_clip=False, zorder=5,
        )
        ax.annotate(
            f"{gray_val:.4f}",
            xy=(tx, thumb_y), xycoords="data",
            xytext=(0, -(box_half_height_pt + 16)), textcoords="offset points",
            ha="center", va="top", fontsize=8, color=gray_color,
            annotation_clip=False, zorder=5,
        )


def add_milestone_markers(ax, ax2, ds, milestone_steps,
                           compliance_color="purple", gray_color="crimson"):
    """Mark the exact compliance / gray-fraction data point at each milestone
    step with a small dot on each curve. The numeric values themselves are
    labeled under the corresponding thumbnail (see add_topology_thumbnails)
    rather than here, so they don't stack on top of each other when the
    curve flattens out near convergence."""
    for target_step in milestone_steps:
        step = nearest_step(ds, target_step)
        c_val = float(ds.loss.sel(step=step).values)
        g_val = float(ds.gray_fraction.sel(step=step).values)

        ax.scatter([step], [c_val], color=compliance_color, s=30, zorder=5,
                   edgecolor="white", linewidth=0.7)
        ax2.scatter([step], [g_val], color=gray_color, s=30, zorder=5,
                    edgecolor="white", linewidth=0.7)


def plot_topology_evolution(
    results, save_path, max_iterations, milestone_steps,
    suptitle="KAN Topology Evolution: Compliance and Gray Fraction Convergence",
    thumb_width_in=0.28, dpi=300, panel_width=6.5, panel_height=5.0, close=True,
):
    """results: list of (label, xarray.Dataset) pairs, each ds must have
    'step', 'loss', 'gray_fraction', and 'design' data."""
    fig, axes = plt.subplots(1, len(results), figsize=(panel_width * len(results), panel_height))
    fig.patch.set_facecolor("white")
    if len(results) == 1:
        axes = [axes]

    for ax, (label, ds) in zip(axes, results):
        ax.set_facecolor("white")
        ax2 = ax.twinx()

        steps = ds.step.values
        compliance = ds.loss.values
        gray_fraction = ds.gray_fraction.values

        line_c, = ax.plot(steps, compliance, color="purple", lw=2.0,
                           linestyle="-", label="Compliance")
        line_g, = ax2.plot(steps, gray_fraction, color="crimson", lw=1.8,
                            linestyle="--", label="Gray Element Fraction")

        ax.set_xlim(0, max_iterations)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Compliance", color="purple")
        ax2.set_ylabel("Gray Element Fraction", color="crimson")
        ax.tick_params(axis="y", labelcolor="purple")
        ax2.tick_params(axis="y", labelcolor="crimson")
        ax.set_title(label, fontsize=13)
        ax.grid(True, alpha=0.3)

        # Fix ax2's limits before inserting thumbnails on ax (primary axis),
        # so the twin axis is not rescaled afterwards.
        ax2.set_ylim(0, max(0.5, float(np.nanmax(gray_fraction)) * 1.3))

        add_milestone_markers(ax, ax2, ds, milestone_steps)
        add_topology_thumbnails(ax, ds, ds.loss, milestone_steps, thumb_width_in=thumb_width_in)

        ax.legend(handles=[line_c, line_g], loc="lower left", frameon=True, fontsize=8.5)

    fig.suptitle(suptitle, fontsize=14, y=1.05)
    fig.tight_layout()
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    if close:
        plt.close(fig)
    return fig
