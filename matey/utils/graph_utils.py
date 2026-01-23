from torch_geometric.utils import to_dense_batch

def graph_to_densenodes(x, batch):
    #data:batched PyG data objects
    #x = data.x          #[N_total, C] or [N_total, T, C]
    #batch = data.batch  #[N_total] graph id per node
    x_dense, mask = to_dense_batch(x, batch, fill_value = 0.0)  
    #x_dense: [B, Max_nodes, C] or [B, Max_nodes ,T, C]
    #mask:    [B, Max_nodes]->True where a real node exists
    return x_dense, mask

def densenodes_to_graphnodes(x_dense, mask):
    #x_dense: [B, Max_nodes, C]or [B, Max_nodes, T, C]
    #mask: [B, Max_nodes]
    x_flat = x_dense[mask]  #[N_total, C] or [N_total, T, C] (only real nodes)
    return x_flat
