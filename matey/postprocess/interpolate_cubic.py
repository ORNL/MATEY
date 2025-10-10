import os
import csv
import numpy as np
from scipy.ndimage import zoom
from tqdm import tqdm

class DataInterpolator:
    def __init__(self, csv_path, dataset_dir, upscale=8):
        """
        Args:
            csv_path (str): Path to e.g. train_data_summary.csv
            dataset_dir (str): Path to dataset folder containing HR/ and LR_8x/
            upscale (int): Upscaling factor (8 for LR_8x)
        """
        self.csv_path = csv_path
        self.dataset_dir = dataset_dir
        self.upscale = upscale
        self.field_names = ['RHO_kgm-3_id', 'UX_ms-1_id', 'UY_ms-1_id', 'UZ_ms-1_id']

        # Deduce subset name (train, val, test, paramvar, forcedhit)
        self.subset_name = os.path.basename(csv_path).replace("_data_summary.csv", "")

        # Define input/output folders
        self.lr_dir = os.path.join(dataset_dir, f"LR_{upscale}x", self.subset_name)
        self.out_dir = os.path.join(dataset_dir, f"LR_{upscale}x_interpolated", self.subset_name)
        os.makedirs(self.out_dir, exist_ok=True)

        # Load CSV metadata
        self.datadict = self._load_csv(csv_path)
        self.num_samples = len(self.datadict['hash'])

        print(f"Interpolating dataset: {self.subset_name}")
        print(f"Input folder: {self.lr_dir}")
        print(f"Output folder: {self.out_dir}")
        print(f"Number of samples: {self.num_samples}")
        print(f"Interpolation factor: {self.upscale}\n")

    # ------------------------------------------------------------------

    def _load_csv(self, path):
        """Reads CSV and returns a dictionary of lists."""
        with open(path, 'r') as f:
            reader = csv.DictReader(f)
            datadict = {col: [] for col in reader.fieldnames}
            for row in reader:
                for col in reader.fieldnames:
                    datadict[col].append(row[col])
        return datadict

    # ------------------------------------------------------------------

    def interpolate_all(self):
        """Perform cubic interpolation on all samples in the subset."""
        SR_ratio = (self.upscale, self.upscale, self.upscale)

        for i in tqdm(range(self.num_samples), desc=f"Interpolating {self.subset_name}"):
            hash_id = self.datadict['hash'][i]

            for scalar in self.field_names:
                # Input/output paths
                lr_path = os.path.join(self.lr_dir, f"{scalar}{hash_id}.dat")
                out_path = os.path.join(self.out_dir, f"{scalar}{hash_id}.dat")

                if not os.path.exists(lr_path):
                    print(f"⚠️ Missing LR file: {lr_path}")
                    continue

                # Read low-res data
                lr_data = np.fromfile(lr_path, dtype=np.float32).reshape(
                    128 // self.upscale, 128 // self.upscale, 128 // self.upscale
                )

                # Tricubic interpolation (order=3)
                hr_interp = zoom(lr_data, zoom=SR_ratio, order=3, mode='nearest')

                # Save interpolated field
                hr_interp.astype(np.float32).tofile(out_path)

        print(f"\n Finished interpolating {self.subset_name}! Saved to: {self.out_dir}")

# ------------------------------------------------------------------

if __name__ == "__main__":
    # Example usage
    dataset_dir = "/pscratch/sd/c/csalah/data/BLASTNET/dataset"
    csv_file = os.path.join(dataset_dir, "../train_data_summary.csv")

    interpolator = DataInterpolator(csv_file, dataset_dir, upscale=8)
    interpolator.interpolate_all()
