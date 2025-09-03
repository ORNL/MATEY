import re
import sys
import matplotlib.pyplot as plt
import numpy as np

if len(sys.argv) < 2:
    print("Usage: python plot_metrics.py <logfile>")
    sys.exit(1)

logfile = sys.argv[1]

# --- Define two separate regex patterns ---
ssim_valid_pattern = re.compile(
    r"SSIM rho:\s*([\d\.eE+-]+),\s*SSIM ux:\s*([\d\.eE+-]+),\s*SSIM uy:\s*([\d\.eE+-]+),\s*SSIM uz:\s*([\d\.eE+-]+)"
)
ssim_interp_pattern = re.compile(
    r"SSIM rho interp:\s*([\d\.eE+-]+),\s*SSIM ux interp:\s*([\d\.eE+-]+),\s*SSIM uy interp:\s*([\d\.eE+-]+),\s*SSIM uz interp:\s*([\d\.eE+-]+)"
)

ssim_valid = []
ssim_interp = []

# --- Read file line by line ---
with open(logfile, "r") as f:
    for line in f:
        m1 = ssim_valid_pattern.search(line)
        if m1:
            ssim_valid.append([float(m1.group(i)) for i in range(1, 5)])
        
        m2 = ssim_interp_pattern.search(line)
        if m2:
            ssim_interp.append([float(m2.group(i)) for i in range(1, 5)])

# --- Convert to numpy arrays ---
ssim_valid = np.array(ssim_valid)   # shape (N, 4)
ssim_interp = np.array(ssim_interp) # shape (N, 4)

# --- Compute averaged SSIM ---
avg_valid = ssim_valid.mean(axis=1) if len(ssim_valid) > 0 else None
avg_interp = ssim_interp.mean(axis=1) if len(ssim_interp) > 0 else None

# --- Compute total averages (single numbers) ---
overall_valid = avg_valid.mean() if avg_valid is not None else None
overall_interp = avg_interp.mean() if avg_interp is not None else None

# --- Plot averaged SSIM ---
plt.figure(figsize=(10, 5))
if avg_valid is not None:
    plt.plot(avg_valid, label="Average SSIM (valid)", linewidth=2)
if avg_interp is not None:
    plt.plot(avg_interp, label="Average SSIM (interp)", linewidth=2, linestyle='--')

plt.xlabel("Batch Index")
plt.ylabel("Average SSIM")
plt.title("Average SSIM per Batch")
plt.ylim(0, 1.05)  # SSIM typically between 0 and 1
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("avg_ssim.png", dpi=200)
plt.show()

# --- Print summary ---
if overall_valid is not None:
    print(f"Overall average SSIM (valid): {overall_valid:.6f}")
if overall_interp is not None:
    print(f"Overall average SSIM (interp): {overall_interp:.6f}")

print("Saved avg_ssim.png")
