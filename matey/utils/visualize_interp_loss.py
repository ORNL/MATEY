import re
import sys
import os
import matplotlib.pyplot as plt

def extract_losses(log_file):
    # Matches: "Train loss: X. Valid loss: Y Valid Interp loss: Z"
    pattern = re.compile(
        r"Train loss:\s*([0-9]+(?:\.[0-9]+)?)\s*\.\s*Valid loss:\s*([0-9]+(?:\.[0-9]+)?)\s*Valid Interp loss:\s*([0-9]+(?:\.[0-9]+)?)"
    )
    train_losses = []
    valid_losses = []
    interp_losses = []

    with open(log_file, 'r') as f:
        for line in f:
            match = pattern.search(line)
            if match:
                train_losses.append(float(match.group(1)))
                valid_losses.append(float(match.group(2)))
                interp_losses.append(float(match.group(3)))
    return train_losses, valid_losses, interp_losses

def plot_losses(train_losses, valid_losses, interp_losses, output_path):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss', marker='o', markersize=3, linewidth=1)
    plt.plot(valid_losses, label='Valid Loss', marker='s', markersize=3, linewidth=1)
    plt.plot(interp_losses, label='Valid Interp Loss', marker='x', markersize=3, linewidth=1)
    plt.xlabel("Logged Samples (Concatenated)")
    plt.ylabel("Loss")
    plt.yscale("log")
    plt.title("Train, Valid, and Valid Interp Loss over Time (Concatenated)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Saved plot to {output_path}")

def main():
    if len(sys.argv) != 2:
        print("Usage: python extract_and_plot_losses.py <directory_with_out_files>")
        sys.exit(1)

    log_dir = sys.argv[1]

    if not os.path.isdir(log_dir):
        print(f"Directory not found: {log_dir}")
        sys.exit(1)

    all_train_losses = []
    all_valid_losses = []
    all_interp_losses = []

    # Process .out files in sorted order
    out_files = sorted(f for f in os.listdir(log_dir) if f.endswith(".out"))
    if not out_files:
        print("No .out files found in directory.")
        sys.exit(1)

    for fname in out_files:
        fpath = os.path.join(log_dir, fname)
        train_losses, valid_losses, interp_losses = extract_losses(fpath)
        if train_losses:
            all_train_losses.extend(train_losses)
            all_valid_losses.extend(valid_losses)
            all_interp_losses.extend(interp_losses)
        else:
            print(f"No losses found in {fname}, skipping.")

    if not all_train_losses:
        print("No losses found in any log files.")
        sys.exit(1)

    output_path = os.path.join(log_dir, "combined_loss_plot.png")
    plot_losses(all_train_losses, all_valid_losses, all_interp_losses, output_path)

if __name__ == "__main__":
    main()
