"""Testing for torchtest"""
#! /usr/bin/env python

# Python imports
import os
import sys

# Module imports
import torch
import pytest

# Local imports
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
import test_networks
from tinytorchtest import tinytorchtest as ttt

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
        train_vars=list(model.named_parameters()),
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
        train_vars=list(model.named_parameters()),
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
        train_vars=list(model.named_parameters()),
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

    # Checks range exception
    with pytest.raises(ttt.RangeException):
        assert ttt.test_suite(
            model,
            loss_fn,
            optim,
            data,
            output_range=(1,2),
            test_inf_vals=True,
            test_nan_vals=True,
            test_output_range=True,
        )

def test_params_dont_change():
    """Tests if parameters don't train"""

    # Sets up the model
    inputs = torch.rand(20,20)
    targets = torch.rand(20,2)
    batch = [inputs, targets]
    model = torch.nn.Linear(20,2)

    # Avoids training the bias term
    params_to_train = [ param[1] for param in model.named_parameters() if param[0] != 'bias']

    ttt.setup(1)

    ttt.assert_vars_same(
        model=model,
        loss_fn=torch.nn.functional.cross_entropy,
        optim=torch.optim.Adam(params_to_train),
        batch=batch,
        device="cpu",
        params=[('bias', model.bias)],
    )

    with pytest.raises(ttt.VariablesChangeException):
        ttt.assert_vars_same(
            model=model,
            loss_fn=torch.nn.functional.cross_entropy,
            optim=torch.optim.Adam(params_to_train),
            batch=batch,
            device="cpu",
            params=list(model.named_parameters()),
        )

def test_nan_exception():

    # Setup test suite
    ttt.setup(1)

    # Model
    layers = [3, 10, 1]
    model = test_networks.SingleArgRegression(layers)

    # Data
    data = [torch.rand(4, 3), torch.rand(4,1)]

    data[0][0, 0] = float('nan')

    # Optimiser
    optim = torch.optim.Adam([p for p in model.parameters() if p.requires_grad])

    # Loss
    loss_fn = torch.nn.MSELoss()

    # run all tests
    with pytest.raises(ttt.NaNTensorException):
        assert ttt.test_suite(
            model,
            loss_fn,
            optim,
            data,
            test_nan_vals=True,
        )

def test_inf_exception():

    # Setup test suite
    ttt.setup(1)

    # Model
    layers = [3, 10, 1]
    model = test_networks.InfModel(layers)

    # Data
    data = [torch.rand(4, 3), torch.rand(4,1)]

    # Optimiser
    optim = torch.optim.Adam([p for p in model.parameters() if p.requires_grad])

    # Loss
    loss_fn = torch.nn.MSELoss()

    # run all tests
    with pytest.raises(ttt.InfTensorException):
        assert ttt.test_suite(
            model,
            loss_fn,
            optim,
            data,
            test_inf_vals=True,
        )



if __name__ == '__main__':
    print("Running tests...")
    test_regression()
    test_regression_multi_args()
    test_regression_unsupervised()
    test_classification()
    test_params_dont_change()
    test_nan_exception()
    test_inf_exception()
    print("Testing complete!")
