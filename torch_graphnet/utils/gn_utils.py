# Copyright 2021 (c) Aalto University - All Rights Reserved
# Author: David Blanco Mulero <david.blancomulero@aalto.fi>
#
import torch

@torch.jit.script
def receiver_nodes_to_edges(nodes:torch.Tensor, receivers:torch.Tensor):
    """
    Gather the receiver nodes from the graph
    Args:
        graph: GNData

    Returns: tensor of shape [E, F_x]

    """
    return nodes[receivers[:, 0], :]

@torch.jit.script
def sender_nodes_to_edges(nodes:torch.Tensor, senders:torch.Tensor):
    """
    Gather the receiver nodes from the graph
    Args:
        graph: GNData

    Returns: tensor of shape [E, F_x]
    """
    return nodes[senders[:, 0], :]

@torch.jit.script
def context_to_edges(edge_attr:torch.Tensor, global_context:torch.Tensor):
    """
    Broadcasts the global features to the edges of the graph
    Args:
        graph: GNData

    Returns: tensor of shape [E, F_u]
    """
    return global_context.repeat(edge_attr.shape[0], 1)

@torch.jit.script
def context_to_nodes(nodes:torch.Tensor, global_context:torch.Tensor):
    """
    Broadcasts the global features to the edges of the graph
    Args:
        graph: GNData

    Returns: tensor of shape [N, F_u]
    """
    return global_context.repeat(nodes.shape[0], 1)


def received_edges_to_node_aggregator(nodes, edge_attr, receivers, reduce:str):
    # out = scatter_mean(out, col, dim=0, dim_size=x.size(0))
    return scatter_sum(edge_attr, receivers, nodes.shape[0])


def sent_edges_to_node_aggregator(nodes, edge_attr, senders, reduce:str):
    return scatter_sum(edge_attr, senders, nodes.shape[0])


@torch.jit.script
def scatter_sum(src: torch.Tensor, idx:torch.Tensor, out_segments: int):
    out = torch.zeros(out_segments, src.shape[1], dtype=torch.float64)
    idxs = idx.expand(src.shape[0], src.shape[1])
    return out.scatter_add(dim=0, index=idxs, src=src)

