"""Testing for torchtest"""
#! /usr/bin/env python

# Python imports
import os
import sys

# Module imports
import torch

# Local imports
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
from tinytorchtest import tinytorchtest as ttt
import test_networks

def test_regression():
    """Tests if a single argument regression trains"""

    # Setup test suite
    ttt.setup(1)

    # Model
    layers = [3, 10, 1]
    model = test_networks.SingleArgRegression(layers)

    # Data
    data = [torch.rand(4, 3), torch.rand(4,1)]

    # Optimiser
    optim = torch.optim.Adam([p for p in model.parameters() if p.requires_grad])

    # Loss
    loss_fn = torch.nn.MSELoss()

    # run all tests
    assert ttt.test_suite(
        model,
        loss_fn,
        optim,
        data,
        test_vars_change=True,
        test_inf_vals=True,
        test_nan_vals=True,
    )

def test_regression_multi_args():
    """Tests if a multi argument regression model trains"""
    # Setup test suite
    ttt.setup(1)

    # Model
    layers = [3, 10, 1]
    model = test_networks.MultiArgRegression(layers)

    # Data
    data = [
        (torch.rand(4, 2), torch.rand(4,1)),    # Inputs
        torch.rand(4,1),                        # Correct outputs
    ]

    # Optimiser
    optim = torch.optim.Adam([p for p in model.parameters() if p.requires_grad])

    # Loss
    loss_fn = torch.nn.MSELoss()

    # run all tests
    assert ttt.test_suite(
        model,
        loss_fn,
        optim,
        data,
        test_vars_change=True,
        test_inf_vals=True,
        test_nan_vals=True,
    )

def test_regression_unsupervised():
    """Tests an unsupervised regression problem"""
    # Setup test suite
    ttt.setup(1)

    # Model
    layers = [3, 10, 1]
    model = test_networks.SingleArgRegression(layers)

    # Data
    data = torch.rand(4, 3)

    # Optimiser
    optim = torch.optim.Adam([p for p in model.parameters() if p.requires_grad])

    # Loss
    def _loss(output):
        return torch.mean(output**2)

    # run all tests
    assert ttt.test_suite(
        model,
        _loss,
        optim,
        data,
        test_vars_change=True,
        test_inf_vals=True,
        test_nan_vals=True,
        supervised=False,
    )

def test_classification():
    """Tests a classification network"""
    # Setup test suite
    ttt.setup(1)

    # Model
    layers = [3, 10, 1]
    model = test_networks.SingleArgClassification(layers)

    # Data
    data = [torch.rand(4, 3), torch.zeros(4, 1)]

    # Optimiser
    optim = torch.optim.Adam([p for p in model.parameters() if p.requires_grad])

    # Loss
    loss_fn = torch.nn.CrossEntropyLoss()

    # Run tests
    assert ttt.test_suite(
        model,
        loss_fn,
        optim,
        data,
        output_range=(0,1),
        test_inf_vals=True,
        test_nan_vals=True,
        test_output_range=True,
    )

if __name__ == '__main__':
    print("Running tests...")
    test_regression()
    test_regression_multi_args()
    test_regression_unsupervised()
    test_classification()
    print("Testing complete!")
