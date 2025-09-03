import os
import re
import sys
import matplotlib.pyplot as plt

if len(sys.argv) < 2:
    print("Usage: python plot_metrics.py <log_directory>")
    sys.exit(1)

log_dir = sys.argv[1]

# Regex patterns
ssim_pattern = re.compile(r"- SSIM:\s*([0-9.]+)")
interp_ssim_pattern = re.compile(r"- Interp SSIM:\s*([0-9.]+)")
nrmse_pattern = re.compile(r"- NRMSE:\s*([0-9.]+)")
interp_nrmse_pattern = re.compile(r"- Interp NRMSE:\s*([0-9.]+)")
epoch_pattern = re.compile(r"(\d+)epochs\.log$")

epochs = []
ssim_values = []
interp_ssim_values = []
nrmse_values = []
interp_nrmse_values = []

# Loop through all .log files
for fname in os.listdir(log_dir):
    if fname.endswith(".log"):
        epoch_match = epoch_pattern.search(fname)
        if epoch_match:
            epoch = int(epoch_match.group(1))
            with open(os.path.join(log_dir, fname), "r") as f:
                content = f.read()
                ssim_match = ssim_pattern.search(content)
                interp_ssim_match = interp_ssim_pattern.search(content)
                nrmse_match = nrmse_pattern.search(content)
                interp_nrmse_match = interp_nrmse_pattern.search(content)
                
                if all([ssim_match, interp_ssim_match, nrmse_match, interp_nrmse_match]):
                    ssim = float(ssim_match.group(1))
                    interp_ssim = float(interp_ssim_match.group(1))
                    nrmse = float(nrmse_match.group(1))
                    interp_nrmse = float(interp_nrmse_match.group(1))
                    
                    epochs.append(epoch)
                    ssim_values.append(ssim)
                    interp_ssim_values.append(interp_ssim)
                    nrmse_values.append(nrmse)
                    interp_nrmse_values.append(interp_nrmse)

if not epochs:
    print(f"No valid log files found in {log_dir}.")
    sys.exit(1)

# Sort by epoch
epochs, ssim_values, interp_ssim_values, nrmse_values, interp_nrmse_values = zip(
    *sorted(zip(epochs, ssim_values, interp_ssim_values, nrmse_values, interp_nrmse_values))
)

# Print extracted values in a table
print(f"{'Epoch':>8} | {'SSIM':>10} | {'Interp SSIM':>12} | {'NRMSE':>10} | {'Interp NRMSE':>12}")
print("-" * 65)
for e, s, i_s, n, i_n in zip(epochs, ssim_values, interp_ssim_values, nrmse_values, interp_nrmse_values):
    print(f"{e:>8} | {s:>10.6f} | {i_s:>12.6f} | {n:>10.6f} | {i_n:>12.6f}")

# Plotting (two subplots)
plt.figure(figsize=(8,8))

# SSIM plot
plt.subplot(2, 1, 1)
plt.plot(epochs, ssim_values, marker='o', label='SSIM')
plt.plot(epochs, interp_ssim_values, marker='s', label='Interp SSIM')
plt.ylabel("SSIM")
plt.title("SSIM vs Epochs")
plt.grid(True)
plt.legend()

# NRMSE plot
plt.subplot(2, 1, 2)
plt.plot(epochs, nrmse_values, marker='o', label='NRMSE')
plt.plot(epochs, interp_nrmse_values, marker='s', label='Interp NRMSE')
plt.xlabel("Epochs")
plt.ylabel("NRMSE")
plt.title("NRMSE vs Epochs")
plt.grid(True)
plt.legend()

plt.tight_layout()

# Save plot
save_path = os.path.join(log_dir, "metrics_vs_epochs.png")
plt.savefig(save_path, dpi=300)
print(f"Plot saved to {save_path}")
