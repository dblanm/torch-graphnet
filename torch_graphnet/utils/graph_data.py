# Copyright 2021 (c) Aalto University - All Rights Reserved
# Author: David Blanco Mulero <david.blancomulero@aalto.fi>
#

import torch
from torch_geometric.data import Data, InMemoryDataset


class GNData(Data):
    def __init__(self, x: torch.Tensor = None, receiver_nodes: torch.Tensor = None,
                 sender_nodes: torch.Tensor = None, edge_index: torch.Tensor = None,
                 edge_attr: torch.Tensor = None, global_context: torch.Tensor = None,
                 y: torch.Tensor = None, pos: torch.Tensor = None,
                 normal: torch.Tensor = None, face: torch.Tensor = None, **kwargs):
        super(GNData, self).__init__(x=x, edge_index=edge_index, edge_attr=edge_attr,
                                     y=y, pos=pos, normal=normal, face=face, kwargs=kwargs)
        # Convert the receivers/senders to int64 and reshape to be a 2-D vector
        self.receivers = receiver_nodes.type(torch.int64).reshape(self.num_edges, 1)
        self.senders = sender_nodes.type(torch.int64).reshape(self.num_edges, 1)
        self.global_context = global_context

    @property
    def nodes(self):
        return self.x

    @property
    def num_global(self):
        return self.global_context.shape[-1]
