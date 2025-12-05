"""
Graph Attention Layer for Sparse Pixel Processing
Based on Graph Attention Networks (GAT) for processing sparse pixel data
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def build_knn_graph(positions: torch.Tensor, k: int = 8) -> torch.Tensor:
    """
    Build K-Nearest Neighbors graph from pixel positions.
    
    Args:
        positions: Pixel positions [N, 2] (u, v coordinates)
        k: Number of nearest neighbors
        
    Returns:
        edge_index: Edge indices [2, num_edges] for graph connectivity
    """
    device = positions.device
    num_nodes = positions.shape[0]
    
    if num_nodes <= k:
        # If fewer nodes than k, connect all nodes
        src = torch.arange(num_nodes, device=device).repeat_interleave(num_nodes - 1)
        dst = torch.cat([
            torch.cat([torch.arange(i, device=device), 
                      torch.arange(i + 1, num_nodes, device=device)])
            for i in range(num_nodes)
        ])
        return torch.stack([src, dst], dim=0)
    
    # Compute pairwise distances
    diff = positions.unsqueeze(0) - positions.unsqueeze(1)  # [N, N, 2]
    distances = torch.norm(diff.float(), dim=-1)  # [N, N]
    
    # Get k nearest neighbors for each node
    _, indices = distances.topk(k + 1, dim=-1, largest=False)  # +1 to exclude self
    indices = indices[:, 1:]  # Remove self-loops
    
    # Create edge index
    src = torch.arange(num_nodes, device=device).repeat_interleave(k)
    dst = indices.reshape(-1)
    
    edge_index = torch.stack([src, dst], dim=0)
    
    return edge_index


class GraphAttentionLayer(nn.Module):
    """
    Graph Attention Layer for processing sparse pixel features.
    
    Uses multi-head attention mechanism to aggregate information
    from neighboring nodes in the graph.
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        heads: int = 8,
        dropout: float = 0.1,
        negative_slope: float = 0.2,
        concat: bool = True,
        bias: bool = True
    ):
        """
        Args:
            in_features: Input feature dimension
            out_features: Output feature dimension
            heads: Number of attention heads
            dropout: Dropout rate
            negative_slope: Negative slope for LeakyReLU
            concat: If True, concatenate head outputs; if False, average them
            bias: If True, add bias to linear transformations
        """
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.heads = heads
        self.dropout = dropout
        self.negative_slope = negative_slope
        self.concat = concat
        
        # Output dimension per head
        if concat:
            assert out_features % heads == 0
            self.head_dim = out_features // heads
        else:
            self.head_dim = out_features
        
        # Linear transformations for query, key, value
        self.W = nn.Linear(in_features, self.head_dim * heads, bias=bias)
        
        # Attention parameters
        self.a_src = nn.Parameter(torch.zeros(1, heads, self.head_dim))
        self.a_dst = nn.Parameter(torch.zeros(1, heads, self.head_dim))
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(out_features if concat else out_features)
        
        # Dropout
        self.dropout_layer = nn.Dropout(dropout)
        
        # Residual projection if dimensions don't match
        self.residual_proj = None
        if in_features != (self.head_dim * heads if concat else out_features):
            self.residual_proj = nn.Linear(in_features, 
                                           self.head_dim * heads if concat else out_features,
                                           bias=False)
        
        self._reset_parameters()
    
    def _reset_parameters(self):
        """Initialize parameters."""
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.a_src)
        nn.init.xavier_uniform_(self.a_dst)
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        return_attention: bool = False
    ) -> torch.Tensor:
        """
        Forward pass of Graph Attention Layer.
        
        Args:
            x: Node features [N, in_features]
            edge_index: Graph connectivity [2, E] where E is number of edges
            return_attention: If True, also return attention weights
            
        Returns:
            Output features [N, out_features]
            (Optional) Attention weights [E, heads]
        """
        num_nodes = x.shape[0]
        
        # Linear transformation
        h = self.W(x)  # [N, heads * head_dim]
        h = h.view(num_nodes, self.heads, self.head_dim)  # [N, heads, head_dim]
        
        # Get source and destination nodes
        src_idx, dst_idx = edge_index  # [E], [E]
        
        # Compute attention scores
        src_features = h[src_idx]  # [E, heads, head_dim]
        dst_features = h[dst_idx]  # [E, heads, head_dim]
        
        # Attention mechanism: a(Wh_i || Wh_j) = a_src * h_i + a_dst * h_j
        alpha = (src_features * self.a_src).sum(dim=-1) + (dst_features * self.a_dst).sum(dim=-1)
        alpha = F.leaky_relu(alpha, negative_slope=self.negative_slope)  # [E, heads]
        
        # Normalize attention scores using softmax over neighbors
        alpha = self._edge_softmax(alpha, dst_idx, num_nodes)  # [E, heads]
        alpha = self.dropout_layer(alpha)
        
        # Aggregate features
        out = self._aggregate(src_features, alpha, dst_idx, num_nodes)  # [N, heads, head_dim]
        
        # Concat or average heads
        if self.concat:
            out = out.view(num_nodes, -1)  # [N, heads * head_dim]
        else:
            out = out.mean(dim=1)  # [N, head_dim]
        
        # Residual connection
        residual = x if self.residual_proj is None else self.residual_proj(x)
        out = out + residual
        
        # Layer normalization
        out = self.layer_norm(out)
        
        if return_attention:
            return out, alpha
        return out
    
    def _edge_softmax(
        self,
        alpha: torch.Tensor,
        dst_idx: torch.Tensor,
        num_nodes: int
    ) -> torch.Tensor:
        """
        Compute softmax over incoming edges for each node.
        
        Args:
            alpha: Attention scores [E, heads]
            dst_idx: Destination node indices [E]
            num_nodes: Number of nodes
            
        Returns:
            Normalized attention weights [E, heads]
        """
        # Subtract max for numerical stability
        alpha_max = torch.zeros(num_nodes, alpha.shape[1], device=alpha.device)
        alpha_max = alpha_max.scatter_reduce(0, dst_idx.unsqueeze(1).expand_as(alpha), 
                                             alpha, reduce='amax', include_self=False)
        alpha = alpha - alpha_max[dst_idx]
        
        # Compute exp
        alpha = alpha.exp()
        
        # Sum over incoming edges
        alpha_sum = torch.zeros(num_nodes, alpha.shape[1], device=alpha.device)
        alpha_sum = alpha_sum.scatter_add(0, dst_idx.unsqueeze(1).expand_as(alpha), alpha)
        
        # Normalize
        alpha = alpha / (alpha_sum[dst_idx] + 1e-8)
        
        return alpha
    
    def _aggregate(
        self,
        src_features: torch.Tensor,
        alpha: torch.Tensor,
        dst_idx: torch.Tensor,
        num_nodes: int
    ) -> torch.Tensor:
        """
        Aggregate source features to destination nodes.
        
        Args:
            src_features: Source node features [E, heads, head_dim]
            alpha: Attention weights [E, heads]
            dst_idx: Destination node indices [E]
            num_nodes: Number of nodes
            
        Returns:
            Aggregated features [N, heads, head_dim]
        """
        # Weight features by attention
        weighted = src_features * alpha.unsqueeze(-1)  # [E, heads, head_dim]
        
        # Aggregate to destination nodes
        out = torch.zeros(num_nodes, self.heads, self.head_dim, device=src_features.device)
        idx = dst_idx.unsqueeze(1).unsqueeze(2).expand_as(weighted)
        out = out.scatter_add(0, idx, weighted)
        
        return out


class MultiHeadGraphAttention(nn.Module):
    """
    Multi-layer Graph Attention Network with skip connections.
    """
    
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        num_layers: int = 6,
        heads: int = 8,
        dropout: float = 0.1
    ):
        """
        Args:
            in_features: Input feature dimension
            hidden_features: Hidden layer dimension
            out_features: Output feature dimension
            num_layers: Number of GAT layers
            heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()
        
        self.layers = nn.ModuleList()
        
        # Input layer
        self.layers.append(
            GraphAttentionLayer(in_features, hidden_features, heads=heads, 
                               dropout=dropout, concat=True)
        )
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.layers.append(
                GraphAttentionLayer(hidden_features, hidden_features, heads=heads,
                                   dropout=dropout, concat=True)
            )
        
        # Output layer
        self.layers.append(
            GraphAttentionLayer(hidden_features, out_features, heads=heads,
                               dropout=dropout, concat=True)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass through multi-layer GAT.
        
        Args:
            x: Node features [N, in_features]
            edge_index: Graph connectivity [2, E]
            
        Returns:
            Output features [N, out_features]
        """
        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index)
            if i < len(self.layers) - 1:
                x = F.elu(x)
                x = self.dropout(x)
        
        return x
