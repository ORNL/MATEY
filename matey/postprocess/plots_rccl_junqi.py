# Import necessary libraries
import os
import pandas as pd
import matplotlib.pyplot as plt
import glob
import re
plt.rcParams.update({'font.size': 20})
plt.rcParams["font.family"] = "Century Gothic"
# Enable inline plotting (if using Jupyter Notebook)

# Specify the directory where log files are stored
log_dir = "./GB/rccl/set-envs"  # Change this to your log directory path

# Compile a regex to extract the operation (op) and number of nodes (nodes) from the file name.
# Example file: "log.Tree.all_gather.n16..3004175"
pattern = re.compile(r"log\.Tree\.(?P<op>\w+)\.n(?P<nodes>\d+)\..*")

# Dictionary to store data frames for each (node_count, op)
data_dict = {}

# Construct the search pattern with the specified log directory
log_pattern = os.path.join(log_dir, "log.Tree.*.n*..*")

# Use glob to find all matching log files in the specified directory
for filename in glob.glob(log_pattern):
    match = pattern.search(os.path.basename(filename))
    if not match:
        continue  # Skip files that do not match the expected pattern
    op = match.group("op")
    nodes = int(match.group("nodes"))
    
    # Read the log file with pandas.
    # The log files have commented header lines (starting with "#") which we skip.
    # We use whitespace as the delimiter and assign column names manually.
    df = pd.read_csv(
        filename,
        comment="#",
        delim_whitespace=True,
        header=None,
        names=["size", "count", "type", "redop", "root",
               "time_out", "algbw_out", "busbw_out", "wrong_out",
               "time_in", "algbw_in", "busbw_in", "wrong_in"]
    )
    
    # Store the data frame in the dictionary keyed by (nodes, op)
    data_dict[(nodes, op)] = df

# Identify the unique node counts from the log files
node_counts = sorted({nodes for (nodes, _) in data_dict.keys()})

fig, axs = plt.subplots(2, 3, figsize=(15, 10))
shapes=["o","d","^","<"]
# For each node count, plot the bus bandwidth (using out-of-place busbw) vs. message size
for isub, nodes in enumerate(node_counts):
    irow=isub//3
    icol=isub%3
    ax=axs[irow,icol]
    #plt.figure(figsize=(6, 6))
    for iop, op in enumerate(["broadcast", "reduce", "all_gather", "reduce_scatter"]):
        key = (nodes, op)
        if key in data_dict:
            df = data_dict[key]
            # Sort the data by message size
            df = df.sort_values("size")
            # Plot using message size on the x-axis and bus bandwidth on the y-axis.
            if iop//2==0:
                ax.plot(df["size"]/1e6, df["busbw_out"], marker=shapes[iop%2], markerfacecolor='none',mew=2, markersize=12, label=op, linewidth=1.5)
            else:
                ax.plot(df["size"]/1e6, df["busbw_out"], marker=shapes[iop%2], mew=2, markersize=12, label=op, linewidth=1.5)
    
    if irow==1:         
        ax.set_xlabel("Message Size (MB)", fontsize=22)
    if icol == 0: 
        ax.set_ylabel("Bandwidth (GB/s)", fontsize=22)
    #plt.xticks(fontsize=16)
    #plt.yticks(fontsize=16)
    if irow==0:
        plt.setp( ax.get_xticklabels(), visible=False)
    ax.set_xscale("log")  # Log scale to accommodate a wide range of message sizes
    ax.set_title(f"{nodes*8} GPUs", fontsize=24)        
    ax.grid(True)
    ax.set_xticks([1,10,100,1000])

axs[1,2].legend(fontsize=18)
plt.savefig(f"rccl-{nodes}.png", bbox_inches='tight')
plt.savefig(f"rccl-{nodes}.pdf", bbox_inches='tight')

plt.show()
