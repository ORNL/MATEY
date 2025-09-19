import re
import sys
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from collections import defaultdict

def extract_losses_and_alpha(log_file):
    # Matches: "Train loss: X. Valid loss: Y Valid Interp loss: Z"
    loss_pattern = re.compile(
        r"Train loss:\s*([0-9]+(?:\.[0-9]+)?)\s*\.\s*Valid loss:\s*([0-9]+(?:\.[0-9]+)?)\s*Valid Interp loss:\s*([0-9]+(?:\.[0-9]+)?)"
    )
    # Matches: "Using grad loss with alpha X"
    alpha_pattern = re.compile(r"Using grad loss with alpha\s*([0-9]+\.[0-9]+)")

    train_losses = []
    valid_losses = []
    interp_losses = []
    alpha_value = None

    with open(log_file, 'r') as f:
        for line in f:
            loss_match = loss_pattern.search(line)
            if loss_match:
                train_losses.append(float(loss_match.group(1)))
                valid_losses.append(float(loss_match.group(2)))
                interp_losses.append(float(loss_match.group(3)))

            alpha_match = alpha_pattern.search(line)
            if alpha_match:
                alpha_value = float(alpha_match.group(1))

    return train_losses, valid_losses, interp_losses, alpha_value

def plot_losses(all_train_losses, all_valid_losses, all_interp_losses, all_alphas, output_path, mode="concat"):
    plt.figure(figsize=(10, 5))

    if mode == "concat":
        # Plot concatenated curves
        plt.plot(all_train_losses, label='Train Loss', marker='o', markersize=3, linewidth=1)
        plt.plot(all_valid_losses, label='Valid Loss', marker='s', markersize=3, linewidth=1)
        plt.plot(all_interp_losses, label='Valid Interp Loss', marker='x', markersize=3, linewidth=1)
        plt.title("Train, Valid, and Valid Interp Loss over Time (Concatenated)")

    elif mode == "separate":
        # Assign unique colors per file or group (train+valid share color)
        num_groups = len(all_train_losses)
        cmap = cm.get_cmap("Set1", num_groups)  # Larger set of distinct colors

        for i, (train, valid, alpha) in enumerate(zip(all_train_losses, all_valid_losses, all_alphas)):
            color = cmap(i)
            if alpha is not None:
                label = f"alpha={alpha}"
            else:
                label = f"Group {i+1} (No Alpha)"

            plt.plot(train, label=f"Train ({label})", linewidth=1, color=color)
            plt.plot(valid, label=f"Valid ({label})", linestyle="--", linewidth=1, color=color)

        plt.title("Train & Valid Loss (Grouped or Separate by Alpha)")

    else:
        raise ValueError("mode must be either 'concat' or 'separate'")

    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.yscale("log")
    plt.grid(True)
    plt.tight_layout()

    # Place legend outside the plot (to the right)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.)
    plt.savefig(output_path, bbox_inches="tight", dpi=200)
    print(f"Saved plot to {output_path}")

def main():
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print("Usage: python extract_and_plot_losses.py <directory_with_out_files> [concat|separate]")
        sys.exit(1)

    log_dir = sys.argv[1]
    mode = sys.argv[2] if len(sys.argv) == 3 else "concat"

    if not os.path.isdir(log_dir):
        print(f"Directory not found: {log_dir}")
        sys.exit(1)

    all_train_losses = []
    all_valid_losses = []
    all_interp_losses = []
    all_alphas = []

    # Process .out files in sorted order
    out_files = sorted(f for f in os.listdir(log_dir) if f.endswith(".out"))
    if not out_files:
        print("No .out files found in directory.")
        sys.exit(1)

    # Storage for grouping files with the same alpha
    grouped_files = defaultdict(lambda: {'train': [], 'valid': [], 'interp': []})
    first_no_alpha = None
    additional_no_alpha_group = {'train': [], 'valid': [], 'interp': []}

    for fname in out_files:
        fpath = os.path.join(log_dir, fname)
        train_losses, valid_losses, interp_losses, alpha_value = extract_losses_and_alpha(fpath)
        if train_losses:
            if alpha_value is not None:  # Group files by alpha
                grouped_files[alpha_value]['train'].extend(train_losses)
                grouped_files[alpha_value]['valid'].extend(valid_losses)
                grouped_files[alpha_value]['interp'].extend(interp_losses)
            else:  # Handle files with no alpha
                if first_no_alpha is None:  # Keep the first no alpha file separate
                    first_no_alpha = {'train': train_losses, 'valid': valid_losses, 'interp': interp_losses}
                else:  # Concatenate all other no alpha files
                    additional_no_alpha_group['train'].extend(train_losses)
                    additional_no_alpha_group['valid'].extend(valid_losses)
                    additional_no_alpha_group['interp'].extend(interp_losses)

    # In "concat" mode, combine everything into one set
    if mode == "concat":
        for alpha, data in grouped_files.items():
            all_train_losses.extend(data['train'])
            all_valid_losses.extend(data['valid'])
            all_interp_losses.extend(data['interp'])
        if first_no_alpha:
            all_train_losses.extend(first_no_alpha['train'])
            all_valid_losses.extend(first_no_alpha['valid'])
            all_interp_losses.extend(first_no_alpha['interp'])
        all_train_losses.extend(additional_no_alpha_group['train'])
        all_valid_losses.extend(additional_no_alpha_group['valid'])
        all_interp_losses.extend(additional_no_alpha_group['interp'])
    elif mode == "separate":
        # Add grouped data (by alpha) to the plotting data
        for alpha, data in grouped_files.items():
            all_train_losses.append(data['train'])
            all_valid_losses.append(data['valid'])
            all_alphas.append(alpha)
        # Add the first no-alpha file (as its own group)
        if first_no_alpha:
            all_train_losses.append(first_no_alpha['train'])
            all_valid_losses.append(first_no_alpha['valid'])
            all_alphas.append(None)  # No alpha for this group
        # Add the concatenated additional no-alpha files (as one group)
        if additional_no_alpha_group['train']:
            all_train_losses.append(additional_no_alpha_group['train'])
            all_valid_losses.append(additional_no_alpha_group['valid'])
            all_alphas.append(None)  # No alpha for this group

    if not all_train_losses:
        print("No losses found in any log files.")
        sys.exit(1)

    output_path = os.path.join(log_dir, f"loss_plot_{mode}.png")
    plot_losses(all_train_losses, all_valid_losses, all_interp_losses, all_alphas, output_path, mode=mode)

if __name__ == "__main__":
    main()