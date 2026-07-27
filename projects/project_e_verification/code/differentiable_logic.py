#!/usr/bin/env python3
"""differentiable_logic.py - ENWI DLR ported to Project E.

Differentiable Logic Reasoner: general-purpose differentiable logic with
fuzzy predicates, AND/OR/NOT/IMPLIES operations, and universal/existential
quantifiers. Replaces our simpler LTL verifier.

Components:
  - SoftLogic: fuzzy logic ops using product t-norm
  - FormulaBuilder: constructs logical formulas from predicates
  - DifferentiableReasoner: evaluates formulas on slot representations

This generalizes our LTL verifier (E:\agi-research\projects\project_e_verification\)
which only supports propositional LTL on discrete states.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional, Callable
from dataclasses import dataclass
import math


class SoftLogic:
    """Differentiable logic operations using product t-norm."""
    @staticmethod
    def and_op(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return a * b
    @staticmethod
    def or_op(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return a + b - a * b
    @staticmethod
    def not_op(a: torch.Tensor) -> torch.Tensor:
        return 1.0 - a
    @staticmethod
    def implies_op(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return SoftLogic.or_op(SoftLogic.not_op(a), b)
    @staticmethod
    def forall(values: torch.Tensor) -> torch.Tensor:
        """Universal quantifier: product of all values."""
        return values.prod(dim=-1)
    @staticmethod
    def exists(values: torch.Tensor) -> torch.Tensor:
        """Existential quantifier: 1 - product(1 - values)."""
        return 1.0 - (1.0 - values).prod(dim=-1)


class Predicate:
    """A fuzzy predicate over objects (e.g., 'red(x)', 'left_of(x, y)')."""
    def __init__(self, name: str, num_args: int, network: Optional[nn.Module] = None):
        self.name = name
        self.num_args = num_args
        self.network = network  # optional NN that computes predicate from slot features

    def __call__(self, *args) -> torch.Tensor:
        """Evaluate predicate on slot arguments. Returns truth value [0,1]."""
        if self.network is not None:
            return torch.sigmoid(self.network(*args))
        else:
            # Default: return 0.5 (no info)
            return torch.full((args[0].shape[0],), 0.5)


class FormulaBuilder:
    """Builds logical formulas from predicates and slots."""
    def __init__(self, slot_dim: int):
        self.slot_dim = slot_dim
        self.predicates: Dict[str, Predicate] = {}

    def add_predicate(self, pred: Predicate):
        self.predicates[pred.name] = pred

    def build_unary(self, pred_name: str, slot: torch.Tensor) -> torch.Tensor:
        """Build phi(x) where phi is unary predicate and x is a slot."""
        if pred_name not in self.predicates:
            raise ValueError(f"Unknown predicate: {pred_name}")
        return self.predicates[pred_name](slot)

    def build_binary(self, pred_name: str, slot1: torch.Tensor, slot2: torch.Tensor) -> torch.Tensor:
        """Build phi(x, y) where phi is binary predicate."""
        if pred_name not in self.predicates:
            raise ValueError(f"Unknown predicate: {pred_name}")
        return self.predicates[pred_name](slot1, slot2)


class DifferentiableReasoner:
    """Evaluates logical formulas on slot representations.

    Inputs: slots (B, K, slot_dim)
    Outputs: truth values for each formula
    """
    def __init__(self, formula_builder: FormulaBuilder):
        self.fb = formula_builder
        self.sl = SoftLogic()

    def exists_color(self, slots: torch.Tensor, color: str) -> torch.Tensor:
        """Query: exists object with given color.

        Formula: exists x. color(x)
        Returns: (B,) truth value
        """
        # Evaluate color(x) for each slot
        pred_values = self.fb.build_unary(color, slots)  # (B, K)
        # exists: 1 - product(1 - values)
        return self.sl.exists(pred_values)

    def forall_color(self, slots: torch.Tensor, color: str) -> torch.Tensor:
        """Query: all objects have given color.

        Formula: forall x. color(x)
        """
        pred_values = self.fb.build_unary(color, slots)  # (B, K)
        return self.sl.forall(pred_values)

    def exists_pair(self, slots: torch.Tensor, rel: str) -> torch.Tensor:
        """Query: exists pair with given relation.

        Formula: exists x, y. rel(x, y)
        """
        B, K, D = slots.shape
        pair_values = []
        for i in range(K):
            for j in range(K):
                if i != j:
                    val = self.fb.build_binary(rel, slots[:, i], slots[:, j])  # (B,)
                    pair_values.append(val)
        return self.sl.exists(torch.stack(pair_values, dim=-1))

    def evaluate_query(self, slots: torch.Tensor, query: str) -> torch.Tensor:
        """Evaluate a simple query string.

        Supported queries:
          - "exists <color>": exists object of color
          - "forall <color>": all objects of color
          - "exists <relation>": exists pair with relation
        """
        parts = query.split()
        if len(parts) != 2:
            raise ValueError(f"Unsupported query: {query}")
        quantifier, predicate = parts
        if quantifier == "exists":
            if predicate in ["red", "blue", "green", "circle", "square"]:
                return self.exists_color(slots, predicate)
            else:
                return self.exists_pair(slots, predicate)
        elif quantifier == "forall":
            return self.forall_color(slots, predicate)
        else:
            raise ValueError(f"Unknown quantifier: {quantifier}")


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(line_buffering=True)
    print("=" * 60)
    print("ENWI DLR (Differentiable Logic Reasoner) — Project E port")
    print("=" * 60)

    # Build a simple example
    slot_dim = 32
    num_slots = 4
    B = 2

    # Define predicates with simple linear networks
    class RedNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Linear(slot_dim, 1)
        def forward(self, x):
            return self.net(x).squeeze(-1)

    class BinaryNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Linear(slot_dim * 2, 1)
        def forward(self, x, y):
            return self.net(torch.cat([x, y], dim=-1)).squeeze(-1)

    red_net = RedNet()
    left_of_net = BinaryNet()

    red_pred = Predicate("red", 1, red_net)
    left_of_pred = Predicate("left_of", 2, left_of_net)

    fb = FormulaBuilder(slot_dim)
    fb.add_predicate(red_pred)
    fb.add_predicate(left_of_pred)

    reasoner = DifferentiableReasoner(fb)

    # Test
    slots = torch.randn(B, num_slots, slot_dim)
    q1 = reasoner.evaluate_query(slots, "exists red")
    q2 = reasoner.evaluate_query(slots, "forall red")
    q3 = reasoner.evaluate_query(slots, "exists left_of")
    print(f"  Query 'exists red': {q1.tolist()}")
    print(f"  Query 'forall red': {q2.tolist()}")
    print(f"  Query 'exists left_of': {q3.tolist()}")
    print()
    print("DLR — ported to Project E, smoke test PASSED")