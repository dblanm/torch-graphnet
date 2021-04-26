# Copyright 2021 (c) Aalto University - All Rights Reserved
# Author: David Blanco Mulero <david.blancomulero@aalto.fi>
#

import graph_nets as gn
import sonnet as snt
import numpy as np
import tensorflow as tf
import torch
import torch_scatter
from torch_geometric.data import Data

from torch_graphnet.utils import receiver_nodes_to_edges, sender_nodes_to_edges, \
    received_edges_to_node_aggregator, sent_edges_to_node_aggregator, context_to_nodes,\
    context_to_edges

from torch_graphnet import EdgeModel, NodeModel, InteractionNetwork

seed = 123
np.random.seed(seed)
tf.random.set_seed(seed)
torch.manual_seed(seed)


class GNData(Data):
    def __init__(self, x: torch.Tensor = None, edge_index: torch.Tensor = None,
                 edge_attr: torch.Tensor = None, global_context: torch.Tensor = None,
                 y: torch.Tensor = None, pos: torch.Tensor = None,
                 normal: torch.Tensor = None, face: torch.Tensor = None, **kwargs):
        super(GNData, self).__init__(x=x, edge_index=edge_index, edge_attr=edge_attr,
                                     y=y, pos=pos, normal=normal, face=face, kwargs=kwargs)
        # Convert the receivers/senders to int64 and reshape to be a 2-D vector
        self.global_context = global_context

    @property
    def nodes(self):
        return self.x

    @property
    def senders(self):
        return self.edge_index[0, :]

    @property
    def receivers(self):
        return self.edge_index[1, :]

    @property
    def num_global(self):
        return self.global_context.shape[-1]

def from_graphstuple_to_gndata(graph: gn.graphs.GraphsTuple):
    # TODO Create from GraphsTuple to GNData
    nodes = torch.from_numpy(graph.nodes.numpy())
    edges = torch.from_numpy(graph.edges.numpy())
    receivers = torch.from_numpy(graph.receivers.numpy()).long()
    senders = torch.from_numpy(graph.senders.numpy()).long()
    globals = torch.from_numpy(graph.globals.numpy())

    edge_index = torch.stack((senders,receivers)).long()

    torch_graph = GNData(x=nodes, edge_index=edge_index.contiguous(),
                         edge_attr=edges, global_context=globals)
    return torch_graph


def get_model_fn(in_size, out_size):
    w_init = np.random.rand(in_size, out_size)
    b_init = np.random.rand(out_size)

    linear_fn_tf = lambda: snt.Linear(output_size=out_size,
                                      w_init=tf.constant_initializer(w_init),
                                      b_init=tf.constant_initializer(b_init))

    linear_fn_torch = torch.nn.Linear(in_size, out_size)
    linear_fn_torch.bias = torch.nn.Parameter(torch.from_numpy(b_init))
    linear_fn_torch.weight = torch.nn.Parameter(torch.from_numpy(w_init.T))

    return linear_fn_tf, linear_fn_torch


def create_graph():
    # Global features for graph 0.
    globals_0 = [1., 2., 3.]

    # Node features for graph 0.
    nodes_0 = [[10., 20., 30.],  # Node 0
               [11., 21., 31.],  # Node 1
               [12., 22., 32.],  # Node 2
               [13., 23., 33.],  # Node 3
               [14., 24., 34.]]  # Node 4

    # Edge features for graph 0.
    edges_0 = [[100., 200.],  # Edge 0
               [101., 201.],  # Edge 1
               [102., 202.],  # Edge 2
               [103., 203.],  # Edge 3
               [104., 204.],  # Edge 4
               [105., 205.]]  # Edge 5

    # The sender and receiver nodes associated with each edge for graph 0.
    senders_0 = [0, 1, 1, 2, 2, 3]  # Index of the sender nodes for the edge i
    receivers_0 = [1, 2, 3, 0, 3, 4]  # Index of the receiver nodes for the edge i

    # Global features for graph 1.
    globals_1 = [1001., 1002., 1003.]

    # Node features for graph 1.
    nodes_1 = [[1010., 1020., 1030.],  # Node 0
               [1012., 1020., 1030.],  # Node 1
               [1013., 1020., 1030.],  # Node 2
               [1014., 1020., 1030.],  # Node 3
               [1015., 1021., 1031.]]  # Node 4

    # Edge features for graph 1.
    edges_1 = [[1100., 1200.],  # Edge 0
               [1101., 1201.],  # Edge 1
               [1102., 1202.],  # Edge 2
               [1102., 1202.],  # Edge 3
               [1103., 1203.]]  # Edge 4

    # The sender and receiver nodes associated with each edge for graph 1.
    senders_1 = [0, 1, 2, 3, 4]
    receivers_1 = [1, 2, 3, 4, 3]

    data_dict_0 = {"globals": np.array(globals_0), "nodes": np.array(nodes_0),
                   "edges": np.array(edges_0),
                   "senders": np.array(senders_0), "receivers": np.array(receivers_0)}
    data_dict_1 = {"globals": np.array(globals_1), "nodes": np.array(nodes_1),
                   "edges": np.array(edges_1),
                   "senders": np.array(senders_1), "receivers": np.array(receivers_1)}
    data_dicts = [data_dict_0, data_dict_1]
    graphs_tuple = gn.utils_tf.data_dicts_to_graphs_tuple(data_dicts)
    return graphs_tuple


def test_gn_utils(graphs_tuple: gn.graphs.GraphsTuple):

    def test_aggregators(graph: GNData, graph_gn):

        reducer = tf.math.unsorted_segment_sum

        sedge_to_nodes = gn.blocks.SentEdgesToNodesAggregator(reducer)
        redge_to_nodes = gn.blocks.ReceivedEdgesToNodesAggregator(reducer)

        nodes_sedge_tf = sedge_to_nodes(graph_gn)
        nodes_redge_tf = redge_to_nodes(graph_gn)
        nodes_sedge = sent_edges_to_node_aggregator(graph.nodes, graph.edge_attr,
                                                    graph.senders, reduce='sum')

        nodes_redge = received_edges_to_node_aggregator(graph.nodes, graph.edge_attr,
                                                        graph.receivers, reduce='sum')

        np.testing.assert_allclose(nodes_sedge_tf.numpy(), nodes_sedge.numpy(),
                                   err_msg="SentEdgesToNodesAggregator does not match")
        np.testing.assert_allclose(nodes_redge_tf.numpy(), nodes_redge.numpy(),
                                   err_msg="ReceivedEdgesToNodesAggregator does not match")

    def test_broadcasts(graph: GNData, graph_gn):
        edge_bcast_rnodes_tf = gn.blocks.broadcast_receiver_nodes_to_edges(graph_gn)
        edge_bcast_snodes_tf = gn.blocks.broadcast_sender_nodes_to_edges(graph_gn)
        edge_bcast_globals_tf = gn.blocks.broadcast_globals_to_edges(graph_gn)

        edge_bcast_rnodes = receiver_nodes_to_edges(graph.nodes, graph.receivers)
        edge_bcast_snodes = sender_nodes_to_edges(graph.nodes, graph.senders)
        edge_bcast_globals = context_to_edges(graph.edge_attr, graph.global_context)

        np.testing.assert_allclose(edge_bcast_rnodes_tf.numpy(), edge_bcast_rnodes.numpy(),
                                   err_msg="Broadcast receiver nodes to edges does not match")
        np.testing.assert_allclose(edge_bcast_snodes_tf.numpy(), edge_bcast_snodes.numpy(),
                                   err_msg="Broadcast sender nodes to edges does not match")
        np.testing.assert_allclose(edge_bcast_globals_tf.numpy(), edge_bcast_globals.numpy(),
                                   err_msg="Broadcast global to edges does not match")

    graph_tf = gn.utils_tf.get_graph(graphs_tuple, index=0)
    graph_torch = from_graphstuple_to_gndata(graph_tf)
    test_aggregators(graph_torch, graph_tf)
    test_broadcasts(graph_torch, graph_tf)


def test_blocks(graphs_tuple: gn.graphs.GraphsTuple):

    def test_edge_block(graph: GNData, graph_gn: gn.graphs.GraphsTuple):
        # Input edge block shape =
        # edge attr shape + num node features (receiver nodes) + num node features (sender nodes)
        in_size = graph.num_edge_features + graph.num_node_features + graph.num_node_features
        out_size = 5

        linear_fn_tf, linear_fn_torch = get_model_fn(in_size, out_size)

        edge_block_tf = gn.blocks.EdgeBlock(edge_model_fn=linear_fn_tf, use_receiver_nodes=True,
                                            use_sender_nodes=True, use_globals=False)
        edge_block_torch = EdgeModel(phi_edge=linear_fn_torch, use_receiver_nodes=True,
                                     use_sender_nodes=True)

        # Compute the output of each model
        out_edge_block_tf = edge_block_tf(graph_gn)
        edges_tf = out_edge_block_tf.edges
        out_edge_block_torch = edge_block_torch(graph.nodes, graph.edge_attr,
                                                graph.edge_index).detach()

        np.testing.assert_allclose(edges_tf.numpy(), out_edge_block_torch.numpy(),
                                   err_msg="Edge block does not match")
    def test_node_block(graph: GNData, graph_gn: gn.graphs.GraphsTuple):
        # Input node block shape = nodes attr shape + num nodes (received edges) + num nodes (sender nodes)
        in_size = graph.num_node_features + graph.num_edge_features + graph.num_edge_features
        out_size = 5  # 5, 7

        linear_fn_tf, linear_fn_torch = get_model_fn(in_size, out_size)

        node_block_tf = gn.blocks.NodeBlock(node_model_fn=linear_fn_tf, use_received_edges=True,
                                            use_sent_edges=True, use_globals=False)
        node_block_torch = NodeModel(phi_node=linear_fn_torch, use_received_edges=True,
                                     use_sent_edges=True, use_context=False)

        # Compute the output of each model
        out_node_block_tf = node_block_tf(graph_gn)
        nodes_tf = out_node_block_tf.nodes
        out_node_block_torch = node_block_torch(graph.nodes, graph.edge_attr,
                                                graph.edge_index).detach()

        np.testing.assert_allclose(nodes_tf.numpy(), out_node_block_torch.numpy(),
                                   err_msg="Node block does not match")

    graph_tf = gn.utils_tf.get_graph(graphs_tuple, index=0)
    graph_torch = from_graphstuple_to_gndata(graph_tf)
    test_edge_block(graph_torch, graph_tf)
    test_node_block(graph_torch, graph_tf)


def test_in(graphs_tuple: gn.graphs.GraphsTuple):

    def test_interaction_network_forward(graph: GNData, graph_gn: gn.graphs.GraphsTuple,
                                 node_fn_tf, node_fn_torch, edge_fn_tf, edge_fn_torch):

        INet_tf = gn.modules.InteractionNetwork(edge_model_fn=edge_fn_tf,
                                                node_model_fn=node_fn_tf)
        INet_torch = InteractionNetwork(phi_edge=edge_fn_torch,
                                        phi_node=node_fn_torch)

        out_graph_tf = INet_tf(graph_gn)
        node_out, edge_out, _ = INet_torch(graph.nodes, graph.edge_attr,
                                        graph.edge_index)

        np.testing.assert_allclose(out_graph_tf.nodes.numpy(), node_out.detach().numpy(),
                                   err_msg="Interaction network Nodes output does not match")
        np.testing.assert_allclose(out_graph_tf.edges.numpy(), edge_out.detach().numpy(),
                                   err_msg="Interaction network Edges output does not match")
        print("Interaction Network forward passed")

        return INet_tf, INet_torch

    def test_interaction_network_backward(inet_tf, inet_torch, graph_in_tf, graph_tgt_tf,
                                          graph_in_torch, graph_tgt_torch):

        with tf.GradientTape() as tape:
            loss_tf = tf.reduce_mean(tf.square(inet_tf(graph_in_tf).nodes - graph_tgt_tf.nodes))
            # loss_value = loss_fn(inet_tf, graph_in_tf, graph_tgt_tf)
            grads = tape.gradient(loss_tf, inet_tf.trainable_variables)
        grad_tf_phi_edge_b = grads[0].numpy()
        grad_tf_phi_edge_w = grads[1].numpy()
        grad_tf_phi_node_b = grads[2].numpy()
        grad_tf_phi_node_w = grads[3].numpy()
        node_out, edge_out, _ = inet_torch(graph_in_torch.nodes, graph_in_torch.edge_attr,
                                              graph_in_torch.edge_index)

        criterion = torch.nn.MSELoss()
        loss = criterion(node_out, graph_tgt_torch.nodes)
        loss.backward()
        grad_torch_phi_node_w = inet_torch.phi_node.weight.grad.detach().numpy().T
        grad_torch_phi_node_b = inet_torch.phi_node.bias.grad.detach().numpy()
        grad_torch_phi_edge_w = inet_torch.phi_edge.weight.grad.detach().numpy().T
        grad_torch_phi_edge_b = inet_torch.phi_edge.bias.grad.detach().numpy()

        np.testing.assert_allclose(grad_tf_phi_node_w, grad_torch_phi_node_w,
                                   err_msg="Phi node weights gradient does not match")
        np.testing.assert_allclose(grad_tf_phi_node_b, grad_torch_phi_node_b,
                                   err_msg="Phi node bias gradient does not match")
        np.testing.assert_allclose(grad_tf_phi_edge_w, grad_torch_phi_edge_w,
                                   err_msg="Phi edge weights gradient does not match")
        np.testing.assert_allclose(grad_tf_phi_edge_b, grad_torch_phi_edge_b,
                                   err_msg="Phi edge bias gradient does not match")
        print("Interaction network gradient passed")




    graph_tf = gn.utils_tf.get_graph(graphs_tuple, index=0)
    graph_tgt_tf = gn.utils_tf.get_graph(graphs_tuple, index=1)
    graph = from_graphstuple_to_gndata(graph_tf)
    graph_tgt = from_graphstuple_to_gndata(graph_tgt_tf)
    edge_size = graph.num_edge_features + graph.num_node_features + graph.num_node_features
    edge_out = 3
    node_size = graph.num_node_features + edge_out
    node_out = 3

    node_fn_tf, node_fn_torch = get_model_fn(node_size, node_out)
    edge_fn_tf, edge_fn_torch = get_model_fn(edge_size, edge_out)

    INet_tf, INet_torch = test_interaction_network_forward(graph, graph_tf, node_fn_tf, node_fn_torch,
                                                           edge_fn_tf, edge_fn_torch)
    test_interaction_network_backward(INet_tf, INet_torch, graph_tf, graph_tgt_tf, graph, graph_tgt)



if __name__ == "__main__":
    graphs_tuple = create_graph()
    test_gn_utils(graphs_tuple)
    test_blocks(graphs_tuple)
    test_in(graphs_tuple)
    print("Everything passed")