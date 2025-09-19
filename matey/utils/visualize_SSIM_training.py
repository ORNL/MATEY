import os
import re
import sys
import matplotlib.pyplot as plt
from collections import defaultdict

if len(sys.argv) < 2:
    print("Usage: python plot_metrics_alpha_models.py <log_directory>")
    sys.exit(1)

log_dir = sys.argv[1]

# Regex patterns for metrics
ssim_pattern = re.compile(r"- SSIM:\s*([0-9.]+)")
nrmse_pattern = re.compile(r"- NRMSE:\s*([0-9.]+)")

# Filename regex
# Matches:
#   InferTest_cubic_100epochs.log                -> old Tiny
#   InferTest_cubic_alpha0_300epochs.log         -> alpha0 Tiny
#   InferTest_cubic_alpha0_S_300epochs.log       -> alpha0 Small
#   InferTest_cubic_alpha099_S_300epochs.log     -> alpha0.99 Small
# Group 1 = alpha, Group 2 = S if present, Group 3 = epoch
fname_pattern = re.compile(r"InferTest_cubic(?:_alpha([0-9]+))?(?:_S)?_(\d+)epochs\.log$")
fname_pattern_with_S = re.compile(r"InferTest_cubic(?:_alpha([0-9]+))?_S_(\d+)epochs\.log$")

# Store data per label
data = defaultdict(lambda: {"epochs": [], "ssim": [], "nrmse": []})

# Loop through all .log files
for fname in os.listdir(log_dir):
    if fname.endswith(".log"):
        # Two possible patterns: with S or without S
        match_S = fname_pattern_with_S.match(fname)
        if match_S:
            alpha_part, epoch_str = match_S.groups()
            model_type = "Small"
        else:
            match = fname_pattern.match(fname)
            if not match:
                continue
            alpha_part, epoch_str = match.groups()
            model_type = "Tiny"

        epoch = int(epoch_str)

        # Build alpha label
        if alpha_part is None:
            alpha_label = f"old ({model_type})"
        else:
            # Convert alpha strings like '07' or '099' into float-ish label
            if len(alpha_part) > 1:
                alpha_val = f"0.{alpha_part[1:]}"
            else:
                alpha_val = alpha_part
            alpha_label = f"alpha{alpha_val} ({model_type})"

        with open(os.path.join(log_dir, fname), "r") as f:
            content = f.read()
            ssim_match = ssim_pattern.search(content)
            nrmse_match = nrmse_pattern.search(content)

            if ssim_match and nrmse_match:
                ssim = float(ssim_match.group(1))
                nrmse = float(nrmse_match.group(1))

                data[alpha_label]["epochs"].append(epoch)
                data[alpha_label]["ssim"].append(ssim)
                data[alpha_label]["nrmse"].append(nrmse)

if not data:
    print(f"No valid log files found in {log_dir}.")
    sys.exit(1)

# --- Plotting SSIM ---
plt.figure(figsize=(9,6))
for alpha_label, d in data.items():
    epochs, ssim_values = zip(*sorted(zip(d["epochs"], d["ssim"])))
    plt.plot(epochs, ssim_values, marker='o', label=alpha_label)

plt.xlabel("Epochs")
plt.ylabel("SSIM")
plt.title("SSIM vs Epochs for different alpha values (Tiny & Small models)")
plt.grid(True)
plt.legend()
plt.tight_layout()

save_path = os.path.join(log_dir, "ssim_vs_epochs_alphas_models.png")
plt.savefig(save_path, dpi=300)
print(f"SSIM plot saved to {save_path}")

# --- Plotting NRMSE ---
plt.figure(figsize=(9,6))
for alpha_label, d in data.items():
    epochs, nrmse_values = zip(*sorted(zip(d["epochs"], d["nrmse"])))
    plt.plot(epochs, nrmse_values, marker='o', label=alpha_label)

plt.xlabel("Epochs")
plt.ylabel("NRMSE")
plt.title("NRMSE vs Epochs for different alpha values (Tiny & Small models)")
plt.grid(True)
plt.legend()
plt.tight_layout()

save_path_nrmse = os.path.join(log_dir, "nrmse_vs_epochs_alphas_models.png")
plt.savefig(save_path_nrmse, dpi=300)
print(f"NRMSE plot saved to {save_path_nrmse}")
