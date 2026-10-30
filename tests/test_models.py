"""Small CPU smoke tests for the public spaNicheGraph model interfaces."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch

from model.create_model import create_model
from model.single_omics import SingleOmicsSpaNicheGraph


def normalized_adjacency(n_nodes: int) -> torch.Tensor:
    adjacency = torch.eye(n_nodes)
    for index in range(n_nodes):
        adjacency[index, (index + 1) % n_nodes] = 1.0
        adjacency[(index + 1) % n_nodes, index] = 1.0
    return adjacency / adjacency.sum(dim=1, keepdim=True)


class ModelSmokeTest(unittest.TestCase):
    @staticmethod
    def paired_args():
        return SimpleNamespace(
            n_input=20,
            n_input1=31,
            n_z=40,
            n_clusters=5,
            gcnii_alpha=0.3,
            gcnii_lambda=0.4,
            sigma=1.0,
            freedom_degree=1.0,
            fusion_mode="concat",
            fusion_hidden_dim=32,
        )

    def test_paired_forward_and_backward(self):
        n_nodes = 12
        args = self.paired_args()
        model = create_model("spanichepair_gcnii", args)
        outputs = model(
            torch.randn(n_nodes, args.n_input),
            torch.randn(n_nodes, args.n_input1),
            normalized_adjacency(n_nodes),
        )
        self.assertEqual(len(outputs), 11)
        self.assertEqual(tuple(outputs[-1].shape), (n_nodes, 2 * args.n_z))
        sum(value.float().mean() for value in outputs if torch.is_tensor(value)).backward()

    def test_ablation_model_names_are_not_exposed(self):
        with self.assertRaises(ValueError):
            create_model("spanichepair_simplegcn", self.paired_args())

    def test_single_omics_forward_and_backward(self):
        n_nodes = 12
        model = SingleOmicsSpaNicheGraph(n_input=50, n_z=40, alpha=0.3, lamda=0.4)
        outputs = model(torch.randn(n_nodes, 50), normalized_adjacency(n_nodes))
        self.assertEqual(len(outputs), 6)
        self.assertEqual(tuple(outputs[-1].shape), (n_nodes, 40))
        sum(value.float().mean() for value in outputs if torch.is_tensor(value)).backward()


if __name__ == "__main__":
    unittest.main()
