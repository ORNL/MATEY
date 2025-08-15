import re
import sys
import matplotlib.pyplot as plt
import numpy as np

if len(sys.argv) < 2:
    print("Usage: python plot_metrics.py <logfile>")
    sys.exit(1)

logfile = sys.argv[1]

# Regex patterns
variance_pattern = re.compile(r"Input variance per channel:\s*tensor\(\[([^\]]+)\]")
ssim_pattern = re.compile(
    r"SSIM rho:\s*([\d\.eE+-]+),\s*SSIM ux:\s*([\d\.eE+-]+),\s*SSIM uy:\s*([\d\.eE+-]+),\s*SSIM uz:\s*([\d\.eE+-]+)"
)
loss_pattern = re.compile(
    r"Valid Loss\s+([\d\.eE+-]+)\s+Interp loss\s+([\d\.eE+-]+)"
)

variances = []
ssim_values = []
valid_losses = []
interp_losses = []
batch_indices = []

batch_counter = 0

with open(logfile, "r") as f:
    for line in f:
        var_match = variance_pattern.search(line)
        if var_match:
            values = [float(v.strip()) for v in var_match.group(1).split(",")]
            variances.append(values)
            batch_indices.append(batch_counter)
            batch_counter += 1
        
        ssim_match = ssim_pattern.search(line)
        if ssim_match:
            ssim_values.append([float(ssim_match.group(i)) for i in range(1, 5)])
        
        loss_match = loss_pattern.search(line)
        if loss_match:
            valid_losses.append(float(loss_match.group(1)))
            interp_losses.append(float(loss_match.group(2)))

# Convert to arrays for easier processing
variances = np.array(variances)  # shape: (batches, channels)
ssim_values = np.array(ssim_values)  # shape: (batches, 4 metrics)

# --- Variance plot (log scale) ---
plt.figure(figsize=(10, 5))
for i in range(variances.shape[1]):
    plt.plot(batch_indices, variances[:, i], label=f"Variance Channel {i}")
plt.xlabel("Batch Index")
plt.ylabel("Variance (log scale)")
plt.yscale("log")
plt.title("Input Variance per Channel (Log Scale)")
plt.legend()
plt.grid(True, which="both", linestyle="--")
plt.tight_layout()
plt.savefig("variance_per_channel_log.png", dpi=200)

# --- SSIM metrics plot ---
plt.figure(figsize=(10, 5))
labels = ["rho", "ux", "uy", "uz"]
for i in range(ssim_values.shape[1]):
    plt.plot(batch_indices, ssim_values[:, i], label=f"SSIM {labels[i]}")
plt.xlabel("Batch Index")
plt.ylabel("SSIM Value")
plt.title("SSIM Metrics per Batch")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("ssim_metrics.png", dpi=200)

# --- Loss plot ---
plt.figure(figsize=(10, 5))
plt.plot(batch_indices[:len(valid_losses)], valid_losses, label="Valid Loss")
plt.plot(batch_indices[:len(interp_losses)], interp_losses, label="Interp Loss")
plt.xlabel("Batch Index")
plt.ylabel("Loss Value")
plt.yscale("log")
plt.title("Validation & Interpolation Loss per Batch")
plt.legend()
plt.grid(True, which="both", linestyle="--")
plt.tight_layout()
plt.savefig("losses.png", dpi=200)

# --- Average variance & SSIM plot ---
avg_variance = variances.mean(axis=1)
avg_ssim = ssim_values.mean(axis=1)

plt.figure(figsize=(10, 5))
plt.plot(batch_indices, avg_variance, label="Average Variance", color="tab:blue")
plt.plot(batch_indices, avg_ssim, label="Average SSIM", color="tab:orange")
plt.xlabel("Batch Index")
plt.ylabel("Value")
plt.yscale("log")
plt.title("Average Variance and Average SSIM per Batch")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("avg_variance_ssim.png", dpi=200)

# --- Scatter plot: Variance vs SSIM ---
plt.figure(figsize=(6, 6))
plt.scatter(avg_variance, avg_ssim, c=batch_indices, cmap="viridis", edgecolor="k")
plt.colorbar(label="Batch Index")
plt.xscale("log")  # variance likely spans orders of magnitude
plt.xlabel("Average Variance (log scale)")
plt.ylabel("Average SSIM")
plt.title("Average SSIM vs Average Variance")
plt.grid(True, which="both", linestyle="--")
plt.tight_layout()
plt.savefig("scatter_variance_vs_ssim.png", dpi=200)

# --- Scatter plot: Loss vs Average Variance ---
plt.figure(figsize=(6, 6))
plt.scatter(avg_variance[:len(valid_losses)], valid_losses, 
            c='tab:blue', label="Valid Loss", alpha=0.7)
plt.scatter(avg_variance[:len(interp_losses)], interp_losses, 
            c='tab:orange', label="Interp Loss", alpha=0.7)
plt.xscale("log")
plt.yscale("log")  # losses also often span orders of magnitude
plt.xlabel("Average Variance (log scale)")
plt.ylabel("Loss (log scale)")
plt.title("Loss vs Average Variance")
plt.legend()
plt.grid(True, which="both", linestyle="--")
plt.tight_layout()
plt.savefig("scatter_loss_vs_variance.png", dpi=200)

print("Plots saved as:")
print("  variance_per_channel_log.png")
print("  ssim_metrics.png")
print("  losses.png")
print("  avg_variance_ssim.png")
print("  scatter_variance_vs_ssim.png")
print("  scatter_loss_vs_variance.png")
