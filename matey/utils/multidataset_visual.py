import re
import argparse
import matplotlib.pyplot as plt
from collections import defaultdict
import math, copy
import numpy as np
import matplotlib
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 18}

matplotlib.rc('font', **font)
def parse_log_file(filepath, dataset_batches, total_norm, total_train_loss, total_valid_loss, epochreal):
    """Parse a single log file and update batch and epoch loss dictionaries."""
    batch_loss_re = re.compile(r"Epoch (\d+) Batch (\d+) Train Loss ([\d\.]+) for (\w+)")
    epoch_loss_re = re.compile(r"Train loss: ([\d\.]+).*?. Valid loss: \s*([\d\.e+-]+|nan+|inf)")
    norm_re = re.compile(r"Pei debugging, total_norm ([\d\.]+) (\w+)")
    with open(filepath, "r") as f:
        for line in f:
            batch_match = batch_loss_re.search(line)
            epoch_match = epoch_loss_re.search(line)
            norm_match = norm_re.search(line)

            if batch_match:
                epoch, batch_idx, loss, dataset = batch_match.groups()
                epoch = int(epoch)
                epoch_=epoch+float(batch_idx)/200.0
                epochreal[dataset].append(epoch_)
                dataset_batches[dataset].append(float(loss))
                #print(filepath, epoch, dataset)

            if norm_match:
                norm, dataset = norm_match.groups()
                total_norm[dataset].append(float(norm))

            if epoch_match:
                #print(filepath, epoch)
                train_loss, valid_loss = epoch_match.groups()
                total_train_loss[epoch].append(float(train_loss))
                total_valid_loss[epoch].append(float(valid_loss))

def moving_average(data, window_size=10):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def plot_dataset_batches(dataset_batches, total_norm, epochreal):
    num_datasets = len(dataset_batches)
    cols = 5
    rows = math.ceil(num_datasets / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4 * rows), squeeze=False)
    fig.suptitle("Batch-wise Training Loss per Dataset", fontsize=16)
    print(epochreal)
    for idx, (dataset, losses) in enumerate(dataset_batches.items()):
        epchs = epochreal[dataset]
        sorted_pairs = sorted(zip(epchs, losses))
        epochlist, losses = zip(*sorted_pairs)
        row = idx // cols
        col = idx % cols
        ax = axes[row][col]
        ax.plot(range(len(losses)), losses, '-+')
        ma_losses = moving_average(losses, window_size=50)
        ax.plot(range(len(ma_losses)), ma_losses, '-', label='Moving Avg')
        ax.set_title(dataset)
        ax.set_xlabel("Accumulated Batch")
        ax.set_ylabel("Loss")
        ax.grid(True)
        ax.set_yscale("log")

    # Hide any unused subplots
    for idx in range(num_datasets, rows * cols):
        fig.delaxes(axes[idx // cols][idx % cols])

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig("./loss_perdataset.png")

    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4 * rows), squeeze=False)
    fig.suptitle("Batch-wise Total Norm per Dataset", fontsize=16)

    for idx, (dataset, losses) in enumerate(total_norm.items()):
        row = idx // cols
        col = idx % cols
        ax = axes[row][col]
        ax.plot(range(len(losses)), losses,'-+')
        ma_losses = moving_average(losses, window_size=50)
        ax.plot(range(len(ma_losses)), ma_losses, '-', label='Moving Avg')
        ax.set_title(dataset)
        ax.set_xlabel("Accumulated Batch")
        ax.set_ylabel("Total Norm")
        ax.set_yscale("log")
        ax.grid(True)

    # Hide any unused subplots
    for idx in range(num_datasets, rows * cols):
        fig.delaxes(axes[idx // cols][idx % cols])

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig("./totalnorm_perdataset.png")

def plot_dataset_batches_selected(dataset_batches, epochreal):
    

    plotdatasetlist={#'swe': "Shallow-water",
    'incompNS': "Incomp NS",
    'diffre2d': "Diffussion-Reaction",
    'compNS': "Comp NS",
    "isotropic1024fine": "Isotropic Homo Turb",
    "MHD256":"MHD Comp Turb",
    'compNS128': "Comp NS 128",
    'compNS512': "Comp NS 512",
    'thermalcollision2d': "Miniweather thermals",
    #"convrsg":"Red Supergiant Convective Envelope",
    "eulerperiodic":"Euler-Riemann",
    #"MHD64":MHD_64,
    "planetswe":"Planet Shallow-water",
    "postneutronstarmerger":"Post-neutron-star-merger",
    "rayleighbenard":"Rayleigh-Benard",
    "rayleightaylor":"Rayleigh-Taylor Turb.",
    "shearflow":"IncompShearFlow",
    "supernova64":"Supernova-explosion64",
    "supernova128":"Supernova-explosion128",
    "turbgravcool":"Turb. Interstellar Medium",
    "turbradlayer2D":"TurbRadiativeLayer2D",
    "turbradlayer3D":"TurbRadiativeLayer3D",
    "viscoelastic": "ViscoelasticInstab."}

    num_datasets = len(plotdatasetlist)
    cols = 5
    rows = math.ceil(num_datasets / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4 * rows), squeeze=False)
    fig.suptitle("Training Loss per Dataset", fontsize=20)
    print(epochreal)
    #for idx, (dataset, losses) in enumerate(dataset_batches.items()):
    for idx, (dataset, datasetname) in enumerate(plotdatasetlist.items()):
        losses = dataset_batches[dataset]
        epchs = epochreal[dataset]
        sorted_pairs = sorted(zip(epchs, losses))
        _, losses = zip(*sorted_pairs)
        row = idx // cols
        col = idx % cols
        ax = axes[row][col]
        ax.plot(range(len(losses)), losses, '-')
        ma_losses = moving_average(losses, window_size=50)
        ax.plot(range(len(ma_losses)), ma_losses, '-', linewidth=1.5)
        ax.set_title(datasetname)
        ax.set_xlabel("Steps")
        #ax.set_ylabel("Loss")
        ax.grid(True)
        ax.set_yscale("log")

    # Hide any unused subplots
    for idx in range(num_datasets, rows * cols):
        fig.delaxes(axes[idx // cols][idx % cols])

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig("./loss_perdataset_selected.png")

def plot_total_losses(total_train_loss, total_valid_loss, prop=1.0):
    epochs = sorted(total_train_loss.keys())
    avg_train_losses = [total_train_loss[ep][-1]  for ep in epochs]
    avg_valid_losses = [total_valid_loss[ep][-1] for ep in epochs]
    print(epochs)
    print(avg_train_losses)
    plt.figure(figsize=(10, 6))
    plt.plot([epoch*prop for epoch in epochs], avg_train_losses, 'r+-', label="Train loss")
    plt.plot([epoch*prop for epoch in epochs], avg_valid_losses, 'bx-', label="Valid loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.yscale("log")
    plt.ylim(top=10.0)
    plt.tight_layout()
    plt.savefig("./loss_total.png")


def main(log_files, prop=1.0):
    dataset_batches = defaultdict(list) 
    total_train_loss = defaultdict(list)
    total_valid_loss = defaultdict(list)
    total_norm = defaultdict(list)
    epochreal = defaultdict(list)
    for file_path in log_files:
        parse_log_file(file_path, dataset_batches, total_norm, total_train_loss, total_valid_loss, epochreal)
    print(epochreal)
    plot_dataset_batches(dataset_batches, total_norm, epochreal)
    plot_dataset_batches_selected(dataset_batches, epochreal)
    plot_total_losses(total_train_loss, total_valid_loss, prop=prop)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot batch and epoch training losses from log files.")
    parser.add_argument("log_files", nargs="+", type=str, help="Path(s) to one or more training log files")
    parser.add_argument("--prop", default=1.0, type=float, help="proportion to real epoch size")
    args = parser.parse_args()

    main(args.log_files, prop=args.prop)

