# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 UT-Battelle, LLC
# This file is part of the MATEY Project.

import torch
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from torch_geometric.data import Data
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from matplotlib.patches import Rectangle

def plot_one_plane_first_channel_with_zoom(
    data,
    plane_tol=1e-8,
    levels=60,
    max_points=None,
    save_path="one_plane_first_channel_zoom.png",
    edge_color="white",
    edge_linewidth=0.12,
    edge_alpha=0.25,
    # zoom window based on data.x
    x_threshold=1.75,
    y_min=-0.5,
    y_max=0.5,
):
    pos = data.pos.detach().cpu()
    ytrue = data.y.detach().cpu()
    ypred = data.ypred.detach().cpu()

    # --- select first z-plane ---
   # z coordinate defines plane
    z = pos[:, 2]

    # first plane = minimum z
    #z0 = z.min()
    z_planes=z.unique()
    z0=z_planes[len(z_planes)//2]
    plane_mask = torch.isclose(z, z0, atol=plane_tol, rtol=0)

    pos2d = pos[plane_mask][:, :2]
    ytrue2d = ytrue[plane_mask, 0]   # first channel only
    ypred2d = ypred[plane_mask, 0]

    if pos2d.shape[0] == 0:
        raise ValueError("No nodes found in first plane.")

    x_zoom_coord = pos2d[:, 0]
    y_zoom_coord = pos2d[:, 1]
    

    # --- build zoom mask from data.x ---
    zoom_mask = (
        (x_zoom_coord > x_threshold) &
        (x_zoom_coord < 2.) &
        (y_zoom_coord > y_min) &
        (y_zoom_coord < y_max)
    )

    if zoom_mask.sum() == 0:
        raise ValueError("No nodes found in the requested zoom region.")

    x_main = pos2d[:, 0].numpy()
    y_main = pos2d[:, 1].numpy()

    triang_main = tri.Triangulation(x_main, y_main)

    z_true = ytrue2d.numpy()
    z_pred = ypred2d.numpy()

    vmin = min(np.nanmin(z_true), np.nanmin(z_pred))
    vmax = max(np.nanmax(z_true), np.nanmax(z_pred))
    levs = np.linspace(vmin, vmax, levels)

    # ---------------- zoom data ----------------
    pos_zoom = pos2d[zoom_mask]
    true_zoom = ytrue2d[zoom_mask].numpy()
    pred_zoom = ypred2d[zoom_mask].numpy()

    xz = pos_zoom[:, 0].numpy()
    yz = pos_zoom[:, 1].numpy()
    triang_zoom = tri.Triangulation(xz, yz)

    # bounding box of zoom region in plotted coordinates
    x1, x2 = xz.min(), xz.max()
    y1, y2 = yz.min(), yz.max()

    #fig, axes = plt.subplots(1, 1, figsize=(14, 6), squeeze=False)
    #ax_pred=axes[0,0]
    fig = plt.figure(figsize=(12, 5))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.8, 1.0], wspace=0.15)

    ax_main = fig.add_subplot(gs[0, 0])
    ax_zoom = fig.add_subplot(gs[0, 1])

    """
    ax_true, ax_pred = axes[0]

    # ---------------- main panels ----------------
    c0 = ax_true.tricontourf(triang_main, z_true, levels=levs)
    ax_true.triplot(triang_main, color=edge_color, linewidth=edge_linewidth, alpha=edge_alpha)
    ax_true.set_title(f"True y[:,0], first plane z={z0.item():.6g}")
    ax_true.set_xlabel("pos_x")
    ax_true.set_ylabel("pos_y")
    ax_true.set_aspect("equal")
    plt.colorbar(c0, ax=ax_true)
    """

    c1 = ax_main.tricontourf(triang_main, z_pred, levels=levs)
    ax_main.triplot(triang_main, color=edge_color, linewidth=edge_linewidth, alpha=edge_alpha)
    ax_main.set_title(f"MATEY Pred e_den, phi={z0.item():.6g}")
    ax_main.set_xlabel("pos_x")
    ax_main.set_ylabel("pos_y")
    ax_main.set_aspect("equal")
    #plt.colorbar(c1, ax=ax_main)
    ax_main.set_axis_off()

    rect = Rectangle(
    (x1, y1), x2 - x1, y2 - y1,
    fill=False, edgecolor="red", linewidth=1.2
    )
    ax_main.add_patch(rect)


    c1 = ax_zoom.tricontourf(triang_zoom, pred_zoom, levels=levs)
    ax_zoom.triplot(triang_zoom, color=edge_color, linewidth=edge_linewidth, alpha=edge_alpha+0.5)
    ax_zoom.set_xlim(x1, x2)
    ax_zoom.set_ylim(y1, y2)
    ax_zoom.set_title("Zoomed region")
    ax_zoom.set_xlabel("pos_x")
    ax_zoom.set_ylabel("pos_y")
    ax_zoom.set_aspect("equal")
    ax_zoom.set_axis_off()
    fig.colorbar(c1, ax=ax_zoom)
 

    plt.tight_layout()
    plt.savefig(save_path, dpi=220, bbox_inches="tight")

def plot_one_plane_y_vs_ypred(
    data,
    channels=(0, 1),
    plane_tol=1e-12,
    max_points=None,
    levels=60,
    varnames=['e_den', 'e_T'],
    save_path="one_plane_y_vs_ypred.png",
    edge_color="white",
    edge_linewidth=0.15,
    edge_alpha=0.35,
):
    pos = data.pos.detach().cpu()
    ytrue = data.y.detach().cpu()
    ypred = data.ypred.detach().cpu()

    # z coordinate defines plane
    z = pos[:, 2]

    # first plane = minimum z
    #z0 = z.min()
    z_planes=z.unique()
    z0=z_planes[len(z_planes)//2]
    mask = torch.isclose(z, z0, atol=plane_tol, rtol=0)

    pos2d = pos[mask][:, :2]
    ytrue2d = ytrue[mask]
    ypred2d = ypred[mask]

    print(f"First plane z = {z0.item()}")
    print(f"Nodes in first plane = {pos2d.shape[0]}")

    if pos2d.shape[0] == 0:
        raise ValueError("No nodes found in first plane.")

    # optional subsampling
    if max_points is not None and pos2d.shape[0] > max_points:
        idx = torch.randperm(pos2d.shape[0])[:max_points]
        pos2d = pos2d[idx]
        ytrue2d = ytrue2d[idx]
        ypred2d = ypred2d[idx]

    x = pos2d[:, 0].numpy()
    y = pos2d[:, 1].numpy()

    triang = tri.Triangulation(x, y)

    nrows = len(channels)
    fig, axes = plt.subplots(nrows, 2, figsize=(12, 5 * nrows), squeeze=False)

    for i, ch in enumerate(channels):
        z_true = ytrue2d[:, ch].numpy()
        z_pred = ypred2d[:, ch].numpy()
        if i==1 and varnames[1]=="e_T":
            #(parallel + 2* perpendiuclar)/3
            z_true = (ytrue2d[:, 2].numpy() + 2*ytrue2d[:, 1].numpy())/3
            z_pred = (ypred2d[:, 2].numpy() + 2*ypred2d[:, 1].numpy())/3

        # same contour levels for fair comparison
        vmin = min(np.nanmin(z_true), np.nanmin(z_pred))
        vmax = max(np.nanmax(z_true), np.nanmax(z_pred))
        levs = np.linspace(vmin, vmax, levels)

        c0 = axes[i, 0].tricontourf(triang, z_true, levels=levs)
        axes[i, 0].triplot(
            triang,
            color=edge_color,
            linewidth=edge_linewidth,
            alpha=edge_alpha,
        )
        axes[i, 0].set_title(f"XGC True {varnames[i]}, phi={z0.item():.6g}")
        axes[i, 0].set_xlabel("x")
        axes[i, 0].set_ylabel("y")
        axes[i, 0].set_aspect("equal")
        plt.colorbar(c0, ax=axes[i, 0])
        axes[i, 0].set_axis_off() # Hides all axes for this Axes object


        c1 = axes[i, 1].tricontourf(triang, z_pred, levels=levs)
        axes[i, 1].triplot(
            triang,
            color=edge_color,
            linewidth=edge_linewidth,
            alpha=edge_alpha,
        )
        axes[i, 1].set_title(f"MATEY Pred {varnames[i]}, phi={z0.item():.6g}")
        axes[i, 1].set_xlabel("x")
        axes[i, 1].set_ylabel("y")
        axes[i, 1].set_aspect("equal")
        plt.colorbar(c1, ax=axes[i, 1])
        axes[i, 1].set_axis_off() # Hides all axes for this Axes object


    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()



def plot_pyg_node_contours(
    data,
    y_channels=None,
    channel_names=None,
    max_points=None,
    levels=50,
    cmap="viridis",
    figsize_per_plot=(6, 5),
    save_path=None,
):
    """
    Plot 2D contour(s) of data.y on node positions data.pos[:, :2].

    Parameters
    ----------
    data : PyG Data or Batch
        Must contain:
          - data.pos: [N, 3] or [N, >=2]
          - data.y:   [N, C] or [N]
    y_channels : list[int] or None
        Which output channels of data.y to plot. If None, plot all.
    channel_names : list[str] or None
        Optional names for channels.
    max_points : int or None
        If set, randomly subsample nodes for faster plotting.
    levels : int
        Number of contour levels.
    cmap : str
        Matplotlib colormap.
    figsize_per_plot : tuple
        Size per subplot.
    save_path : str or None
        If provided, save figure to this file.
    """

    # Move tensors to CPU and convert to numpy
    pos = data.pos.detach().cpu()
    y = data.y.detach().cpu()

    # Handle y shape [N] -> [N, 1]
    if y.ndim == 1:
        y = y.unsqueeze(-1)

    # Optional subsampling for very large graphs
    if max_points is not None and pos.shape[0] > max_points:
        idx = torch.randperm(pos.shape[0])[:max_points]
        pos = pos[idx]
        y = y[idx]

    xcoord = pos[:, 0].numpy()
    ycoord = pos[:, 1].numpy()
    values = y.numpy()

    n_channels = values.shape[1]

    if y_channels is None:
        y_channels = list(range(n_channels))

    n_plots = len(y_channels)

    if channel_names is None:
        channel_names = [f"y[{i}]" for i in range(n_channels)]

    # Layout
    ncols = min(3, n_plots)
    nrows = (n_plots + ncols - 1) // ncols

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(figsize_per_plot[0] * ncols, figsize_per_plot[1] * nrows),
        squeeze=False
    )
    axes = axes.flatten()

    # Triangulation for unstructured node positions
    triang = tri.Triangulation(xcoord, ycoord)

    for ax, ch in zip(axes, y_channels):
        z = values[:, ch]

        contour = ax.tricontourf(triang, z, levels=levels, cmap=cmap)
        ax.set_title(channel_names[ch] if ch < len(channel_names) else f"y[{ch}]")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect("equal")
        fig.colorbar(contour, ax=ax)

    # Hide unused axes
    for ax in axes[n_plots:]:
        ax.axis("off")

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")

    plt.show()


# ---------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------
if __name__ == "__main__":
    ##torch.save({"inp":inp, "target": tar, "output":output}, f"matey_{case}_leadtime_{leadtime[0].item()}.pt")
    data = torch.load("/lustre/orion/lrn037/scratch/zhangp/fy25/github/coderelease/MATEY/examples/matey_XGC_leadtime_5.0.pt", weights_only=False)
    graph = data["inp"]
    graph.ypred=data["output"]
    channel_names =['e_den', 'e_T_perp', 'e_T_para', 'e_u_para', 'i_T_perp','i_T_para', 'i_u_para', 'dpot']

    print(graph)

    plot_one_plane_first_channel_with_zoom(
        graph,
        plane_tol=1e-8,
        levels=60,
        save_path="one_plane_first_channel_with_zoom.png",
        edge_color="white",
        edge_linewidth=0.12,
        edge_alpha=0.25,
        x_threshold=1.9,
        y_min=-0.2,
        y_max=0.2,
    )

    plot_one_plane_y_vs_ypred(
    graph,
    channels=(0, 1),
    plane_tol=1e-8,
    levels=60,
    varnames=['e_den', 'e_T'],
    save_path="one_plane_comparison.png",
    )
    # Plot all channels
    plot_pyg_node_contours(
        graph,
        y_channels=list(range(graph.y.shape[1])),
        channel_names=channel_names,
        max_points=200000,   # reduce if plotting is too slow
        levels=50,
        cmap="viridis",
        save_path="pyg_y_contours.png",
    )