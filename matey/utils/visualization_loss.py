#!/usr/bin/env python3
import re
import glob
import argparse
from pathlib import Path
import matplotlib.pyplot as plt

def parse_metrics_from_file(path):
    """
    Parses a single log file and returns three dicts, each mapping epoch -> value:
      - losses[epoch] = (train_loss, val_loss)
      - times[epoch]  = epoch_time (in seconds)
    It looks for lines of the form:
      Time taken for epoch <N> is <T> sec
    followed (not necessarily immediately) by:
      Train loss: <L>. Valid loss: <V>
    """
    # Regex to capture "Time taken for epoch 42 is 273.4864 sec"
    time_re  = re.compile(r'Time taken for epoch\s+(\d+)\s+is\s+([0-9.]+)\s+sec')
    # Regex to capture "Train loss: 0.3148. Valid loss: 0.3441"
    loss_re  = re.compile(r'Train loss:\s*([0-9.eE+-]+)\.\s*Valid loss:\s*([0-9.eE+-]+)')
    # We'll store:
    #   times_map[epoch]  = time_in_sec (float)
    #   losses_map[epoch] = (train_loss, val_loss)
    times_map = {}
    losses_map = {}
    curr_epoch_for_loss = None

    with open(path, 'r') as f:
        for line in f:
            # 1) Look for the time line first:
            tm = time_re.search(line)
            if tm:
                epoch = int(tm.group(1))
                tval  = float(tm.group(2))
                times_map[epoch] = tval
                # After capturing time, the next loss line for the same epoch often follows
                curr_epoch_for_loss = epoch
                continue

            # 2) Look for the loss line if we've just seen a time for that epoch:
            lm = loss_re.search(line)
            if lm and curr_epoch_for_loss is not None:
                train_loss = float(lm.group(1))
                val_loss   = float(lm.group(2))
                losses_map[curr_epoch_for_loss] = (train_loss, val_loss)
                curr_epoch_for_loss = None

    return losses_map, times_map

def collect_case_metrics(patterns):
    """
    Given a list of glob-patterns, find all matching files, parse each,
    and merge their epoch→metrics maps. Later files override earlier epochs.
    Returns three parallel lists (sorted by epoch):
      epochs, train_list, val_list, time_list
    """
    merged_losses = {}  # epoch -> (train, val)
    merged_times  = {}  # epoch -> time_in_sec
    any_file = False

    for pat in patterns:
        for filename in sorted(glob.glob(pat)):
            any_file = True
            losses_map, times_map = parse_metrics_from_file(filename)
            merged_losses.update(losses_map)
            merged_times.update(times_map)
            #print(pat, filename, losses_map,merged_losses)

    if not any_file:
        raise FileNotFoundError(f"No files match any of: {patterns}")

    epochs = sorted(merged_losses)
    train_list = [merged_losses[e][0] for e in epochs]
    val_list   = [merged_losses[e][1] for e in epochs]
    time_list  = [merged_times.get(e, float('nan')) for e in epochs]

    return epochs, train_list, val_list, time_list

def plot_curves(all_epochs, all_trains, all_vals, case_names, outname=""):
    markers = ['o', 's', '^', 'D', 'v', '*', 'x', 'P', 'h']
    colors = ['blue', 'green', 'red', 'orange', 'purple', 'brown', 'pink', 'gray', 'cyan']

    # Train losses
    fig, axs=plt.subplots(1,2, figsize=(16,5))
    for i, name in enumerate(case_names):
        marker = markers[i % len(markers)]
        color = colors[i % len(colors)]
        axs[0].plot(all_epochs[name], all_trains[name],
                marker=marker, linestyle='-', mfc='none',
                color=color, label=f'{name} Train')
        #axs[0].plot(all_epochs[name], all_trains[name],
        #         marker='o', linestyle='-', mfc='none',
        #         label=f'{name} Train')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Train Loss')
    axs[0].set_title('Train Loss vs. Epoch')
    axs[0].legend()
    axs[0].grid(True)
    axs[0].set_yscale('log')
    ylim=axs[0].get_ylim()

    # Validation losses
    for i, name in enumerate(case_names):
        marker = markers[i % len(markers)]
        color = colors[i % len(colors)]
        axs[1].plot(all_epochs[name], all_vals[name],
                marker=marker, linestyle='--', mfc='none',
        #axs[1].plot(all_epochs[name], all_vals[name],
        #         marker='s', linestyle='--', mfc='none',
                 label=f'{name} Val')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Validation Loss')
    axs[1].set_title('Validation Loss vs. Epoch')
    axs[1].legend()
    # axs[1].set_ylim(ylim)
    axs[1].grid(True)
    axs[1].set_yscale('log')


    plt.tight_layout()
    plt.savefig(f"./loss_curves_finalcomp{outname}.png", dpi=300)

def plot_time_curves(all_epochs, all_times, case_names, outname=""):
    plt.figure(figsize=(8,5))
    for name in case_names:
        plt.plot(
            all_epochs[name],
            all_times[name],
            marker='^', linestyle='-.',
            label=f'{name} Time'
        )
    plt.xlabel('Epoch')
    plt.ylabel('Time per Epoch (sec)')
    plt.title('Epoch Time vs. Epoch')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"./time_curves_finalcomp{outname}.png", dpi=300)

def main():
    parser = argparse.ArgumentParser(
        description="Compare train/val loss curves across multiple cases"
    )
    parser.add_argument(
        '--case', '-c',
        action='append',
        nargs='+',              
        metavar=('NAME', 'PATTERNS'),
        help="Specify a case: NAME and a list of glob PATTERNs matching its log files"
    )
    parser.add_argument("--outname", default="", help="string in output case")

    args = parser.parse_args()
    print(args)

    if not args.case or len(args.case) < 2:
        parser.error("Please provide at least two --case NAME PATTERN [PATTERN ...] entries")

    all_epochs = {}
    all_trains = {}
    all_vals   = {}
    case_names = []

    all_epochs = {}
    all_trains = {}
    all_vals   = {}
    all_times  = {}
    case_names = []

    for case_spec in args.case:
        name = case_spec[0]
        patterns = case_spec[1:]
        print(name, patterns)
        try:
            epochs, trains, vals, times = collect_case_metrics(patterns)
        except FileNotFoundError as e:
            parser.error(str(e))

        all_epochs[name] = epochs
        all_trains[name] = trains
        all_vals[name]   = vals
        all_times[name]  = times
        case_names.append(name)

    plot_curves(all_epochs, all_trains, all_vals, case_names, outname=args.outname)
    plot_time_curves(all_epochs, all_times, case_names, outname=args.outname)


if __name__ == '__main__':
    main()
