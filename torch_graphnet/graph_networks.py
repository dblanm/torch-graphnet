# Copyright 2021 (c) Aalto University - All Rights Reserved
# Author: David Blanco Mulero <david.blancomulero@aalto.fi>
#
# This code is based on the PyTorch Geometric MetaLayer
# https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html?highlight=graph%20network#models
# and the Graph Networks: Relational inductive biases, deep learning, and graph networks
# https://github.com/deepmind/graph_nets

import torch
from torch import nn
from torch_scatter import scatter_mean
from torch_geometric.nn import MetaLayer

from torch_graphnet.utils import sender_nodes_to_edges, receiver_nodes_to_edges, \
    context_to_edges, context_to_nodes, received_edges_to_node_aggregator, \
    sent_edges_to_node_aggregator


class EdgeModel(nn.Module):
    def __init__(self, phi_edge: nn.Module, use_receiver_nodes=True,
                 use_sender_nodes=True, use_context=False):
        super(EdgeModel, self).__init__()

        self.use_receiver_nodes = use_receiver_nodes
        self.use_sender_nodes = use_sender_nodes
        self.use_context = use_context
        self.phi_edge = phi_edge

    def forward(self, nodes, edge_attr, senders, receivers, context=None):
        """
        Args:
           src: [E, F_x], where E is the number of edges.
            dest: [E, F_x], where E is the number of edges.
            edge_attr: [E, F_e]
            u: [B, F_u], where B is the number of graphs.
            batch: [E] with max entry B - 1.

        Returns: updated edges [E]

        """
        if self.use_context and _context is None:
            raise ValueError("EdgeModel use_globals set to True and globals not provided")
        x = [edge_attr]
        if self.use_receiver_nodes: x.append(receiver_nodes_to_edges(nodes, receivers))
        if self.use_sender_nodes: x.append(sender_nodes_to_edges(nodes, senders))
        if self.use_context: x.append(context_to_edges(edge_attr, _context))

        x = torch.cat(x, dim=-1)

        return self.phi_edge(x)


class NodeModel(torch.nn.Module):
    def __init__(self, phi_node: nn.Module, use_received_edges=True,
                 use_sent_edges=False, use_context=False, reduce:str= 'sum'):
        super(NodeModel, self).__init__()
        self.phi_node = phi_node
        self.use_received_edges = use_received_edges
        self.use_sent_edges = use_sent_edges
        self.use_context = use_context
        self.reduce = reduce

    def forward(self, nodes, edge_attr, senders, receivers, context=None):
    # def forward(self, graph:GNData):
        """
        Args:
            x: [N, F_x], where N is the number of nodes.
            edge_index: [2, E] with max entry N - 1.
            edge_attr: [E, F_e]
            u: [B, F_u]
        Returns:
        """
        if self.use_context and context is None:
            raise ValueError("EdgeModel use_globals set to True and globals not provided")
        x = []
        if self.use_received_edges:
            x.append(received_edges_to_node_aggregator(nodes, edge_attr, receivers, reduce=self.reduce))
        if self.use_sent_edges:
            x.append(sent_edges_to_node_aggregator(nodes, edge_attr, senders, reduce=self.reduce))
        if self.use_context: x.append(context_to_nodes(nodes, context))

        x.append(nodes)
        x = torch.cat(x, dim=1)

        return self.phi_node(x)


class GraphNetwork(nn.Module):
    def __init__(self, phi_edge:nn.Module, phi_node:nn.Module, phi_context:nn.Module):
        super(GraphNetwork, self).__init__()
        self.phi_edge = phi_edge
        self.phi_node = phi_node
        self.phi_global = phi_context


    def forward(self, nodes, edge_attr, senders, receivers, context=None):
        raise NotImplementedError


class InteractionNetwork(GraphNetwork):

    def __init__(self, phi_edge: nn.Module, phi_node: nn.Module):
        super(InteractionNetwork, self).__init__(phi_edge=phi_edge, phi_node=phi_node, phi_context=None)
        # TODO Assert output of phi edge is input of phi node
        # TODO Assert input of phi edge and phi node is correct (ask for shapes of nodes and edge_attr.

        self.edge_model = EdgeModel(phi_edge=self.phi_edge, use_receiver_nodes=True,
                                    use_sender_nodes=True, use_context=False)
        self.node_model = NodeModel(phi_node=self.phi_node, use_received_edges=True,
                                    use_sent_edges=False, use_context=False)

    def forward(self, nodes, edge_attr, senders, receivers):

        edge_out = self.edge_model(nodes, edge_attr, senders, receivers)
        node_out = self.node_model(nodes, edge_out, senders, receivers)

        return node_out, edge_out


class GraphIndependent(GraphNetwork):
    def __init__(self, phi_edge:nn.Module=None, phi_node:nn.Module=None, phi_context:nn.Module=None):
        if phi_edge is None:
            phi_edge = lambda x: x
        if phi_node is None:
            phi_node = lambda x: x
        if phi_context is None:
            phi_context = lambda x: x
        super(GraphIndependent, self).__init__(phi_edge, phi_node, phi_context)

    def forward(self, nodes, edge_attr, senders, receivers, context=None):

        node_out = self.phi_node(nodes)
        edge_out = self.phi_edge(edge_attr)
        context_out = self.phi_global(context)

        return node_out, edge_out, context_out
#
# size_in = self._get_input_size(graph)
# # TODO We could leave the phi edge as an input
# # However we would need a requirement of the input size
# # Without an example input graph, that would not be possible
# # We can create this and specify it in the MetaLayer
# self.phi_edge = nn.Linear(size_in, output_size)
#
# def _get_input_size(self, graph:GNData):
#     size_in = graph.num_edges
#     if self.use_receiver_nodes: size_in += graph.num_nodes
#     if self.use_sender_nodes: size_in += graph.num_nodes
#     if self.use_globals: size_in += graph.num_global
#
#     return size_in