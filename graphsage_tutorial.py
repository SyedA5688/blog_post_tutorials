"""
This file is a from-scratch Pytorch implementation of the GraphSAGE message-passing 
operator, for tutorial purposes.

@author: Syed Rizvi (syed.rizvi@yale.edu)

The actual GraphSAGE implementation in Pytorch Geometric can be found at:
https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.SAGEConv.html
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data


class GraphSAGELayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        # Define linear layers parameterizing neighbors and self-loops
        self.lin_neighbors = nn.Linear(in_dim, out_dim, bias=True)
        self.lin_self = nn.Linear(in_dim, out_dim, bias=True)

    def message_passing(self, x: torch.Tensor, edge_index: torch.Tensor):
        """
        This function is responsible for passing messages between nodes according to the edges 
        in 'edge_index'.
        - Messages from the source --> destination node consist of the source nodes feature vector.
        - Sum aggregation is used to aggregate incoming messages from neighbors.

        Arguments:
            x: torch Tensor of node representations, shape [num_nodes, hidden_size], dtype torch.float32
            edge_index: torch Tensor of graph connectivity information, shape [2, num_edges], dtype torch.int64
        Returns:
            neigh_embeds: torch Tensor of aggregated neighbor embeddings, shape [num_nodes, hidden_size], dtype torch.float32
        """
        src_node_indices = edge_index[0, :]  # shape [num_edges]
        dst_node_indices = edge_index[1, :]  # shape [num_edges]
        src_node_feats = x[src_node_indices]  # shape [num_edges, hidden_size]

        # Mean aggregation
        neighbor_sum_aggregations = []
        for dst_node_idx in range(x.shape[0]):  # loop over destination nodes
            # find incoming edges, get incoming messages from source nodes
            incoming_edge_indices = torch.where(dst_node_indices == dst_node_idx)[0]  # find incoming edges
            incoming_messages = src_node_feats[incoming_edge_indices]  # shape [num_incoming_edges, hidden_size]

            # Aggregate step: sum messages from neighbors (if > 1 neighbors)
            incoming_messages_summed = incoming_messages.sum(dim=0) if incoming_messages.shape[0] > 1 else incoming_messages
            neighbor_sum_aggregations.append(incoming_messages_summed)
        
        neigh_embeds = torch.stack(neighbor_sum_aggregations)  # [num_nodes, hidden_size]
        return neigh_embeds

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        """
        Implementation for one message-passing iteration for GraphSAGE.

        Arguments:
            x: torch Tensor of node representations, shape [num_nodes, hidden_size], dtype torch.float32
            edge_index: torch Tensor of graph connectivity information, shape [2, num_edges], dtype torch.int64
        Returns:
            out: torch Tensor of updated node representations, shape [num_nodes, hidden_size], dtype torch.float32
        """
        x_message_passing, x_self = x, x  # duplicate variables pointing to node features
        neigh_embeds = self.message_passing(x_message_passing, edge_index)
        neigh_embeds = self.lin_neighbors(neigh_embeds)
        
        x_self = self.lin_self(x_self)
        # Sum Concatenation
        out = neigh_embeds + x_self
        return out
        


class GraphSAGEModel(nn.Module):
    def __init__(self, in_features: int, hidden_size: int, out_features: int, dropout: int = 0.1):
        super().__init__()
        self.input_proj = nn.Linear(in_features, hidden_size, bias=True)

        self.conv1 = GraphSAGELayer(in_dim=hidden_size, out_dim=hidden_size)
        self.act1 = nn.ReLU()
        self.drop1 = nn.Dropout(p=dropout)
        
        self.lin_out = nn.Linear(hidden_size, out_features, bias=True)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        """
        Forward pass implementation of 1-layer GraphSAGE model.
        
        Arguments:
            x: torch Tensor of input node features, shape [num_nodes, num_features], dtype torch.float32
            edge_index: torch Tensor of graph connectivity information, shape [2, num_edges], dtype torch.int64
        """
        x = self.input_proj(x)  # Input projection: [num_nodes, num_features] --> [num_nodes, hidden_size]

        x = self.conv1(x, edge_index)  # Message-passing
        x = self.act1(x)
        x = self.drop1(x)
        
        x = self.lin_out(x)
        return F.log_softmax(x, dim=-1)  # softmax over last dim for classification


def adj_matrix_to_sparse_edge_index(adj_matr: torch.Tensor):
    """
    This function takes a square binary adjacency matrix, and returns an edge list representation
    containing source and destination node indices for each edge.

    Arguments:
        adj_matr: torch Tensor of adjacency information, shape [num_nodes, num_nodes], dtype torch.int64
    Returns:
        edge_index: torch Tensor of shape [2, num_edges], dtype torch.int64
    """
    src_list = []
    dst_list = []
    for row_idx in range(adj_matr.shape[0]):
        for col_idx in range(adj_matr.shape[1]):
            if adj_matr[row_idx, col_idx].item() > 0.0:
                src_list.append(row_idx)
                dst_list.append(col_idx)
    return torch.tensor([src_list, dst_list], dtype=torch.int64)  # [2, num_edges]


if __name__ == "__main__":
    # Define input node feature matrix and adjacency matrix
    # atomic number, atomic mass, charge, and number of bonds
    input_node_feature_matrix = torch.tensor([
        [1.0, 1.0078, 1, 2],
        [1.0, 1.0078, 1, 4],
        [6.0, 12.011, -1, 3],
        [1.0, 1.0078, 0, 4],
        [6.0, 12.011, -1, 3],
        [1.0, 1.0078, 1, 2],
    ], dtype=torch.float32)

    binary_adjacency_matrix = torch.tensor([
        [1, 1, 0, 0, 0, 0],
        [1, 1, 1, 0, 1, 0],
        [0, 1, 1, 1, 0, 0],
        [0, 0, 1, 1, 1, 1],
        [0, 1, 0, 1, 1, 0],
        [0, 0, 0, 1, 0, 1],
    ], dtype=torch.int64)
    edge_index = adj_matrix_to_sparse_edge_index(binary_adjacency_matrix)
    print("input_node_feature_matrix:", input_node_feature_matrix.shape)
    print("binary_adjacency_matrix:", binary_adjacency_matrix.shape)

    # Define GraphSAGE model
    model = GraphSAGEModel(
        in_features=4,  # 4 input features per node
        hidden_size=16,  # 16-dimensional latent vectors
        out_features=2  # 2 classes of nodes in our example: Carbon and Hydrogen
    )
    print("Model:\n")
    print(model)

    # Forward pass & loss calculation for node classification
    output = model(x=input_node_feature_matrix, edge_index=edge_index)
    atom_labels = torch.tensor([0, 0, 1, 0, 1, 0], dtype=torch.int64)  # 0 = Hydrogen, 1 = Carbon
    loss = F.nll_loss(output, target=atom_labels)
    print("Loss value: {:.5f}".format(loss.item()))
    print("Script finished, exiting...")
