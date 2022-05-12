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

    # Sets random seed
    torch.manual_seed(1)

    # Model
    layers = [3, 10, 1]
    model = test_networks.SingleArgRegression(layers)

    # Data
    batch = [torch.rand(4, 3), torch.rand(4,1)]

    # Optimiser
    optim = torch.optim.Adam([p for p in model.parameters() if p.requires_grad])

    # Loss
    loss_fn = torch.nn.MSELoss()

    # Setup test suite
    test = ttt.TinyTorchTest(model, loss_fn, optim, batch)

    test.test(
        train_vars=list(model.named_parameters()),
        test_vars_change=True,
        test_inf_vals=True,
        test_nan_vals=True,
    )


def test_regression_multi_args():
    """Tests if a multi argument regression model trains"""

    # Sets random seed
    torch.manual_seed(1)

    # Model
    layers = [3, 10, 1]
    model = test_networks.MultiArgRegression(layers)

    # Data
    batch = [
        (torch.rand(4, 2), torch.rand(4,1)),    # Inputs
        torch.rand(4,1),                        # Correct outputs
    ]

    # Optimiser
    optim = torch.optim.Adam([p for p in model.parameters() if p.requires_grad])

    # Loss
    loss_fn = torch.nn.MSELoss()

    # Setup test suite
    test = ttt.TinyTorchTest(model, loss_fn, optim, batch)

    test.test(
        train_vars=list(model.named_parameters()),
        test_vars_change=True,
        test_inf_vals=True,
        test_nan_vals=True,
    )

def test_regression_unsupervised():
    """Tests an unsupervised regression problem"""

    # Sets random seed
    torch.manual_seed(1)

    # Model
    layers = [3, 10, 1]
    model = test_networks.SingleArgRegression(layers)

    # Data
    batch = torch.rand(4, 3)

    # Optimiser
    optim = torch.optim.Adam([p for p in model.parameters() if p.requires_grad])

    # Loss
    def _loss(output):
        return torch.mean(output**2)

    # Setup test suite
    test = ttt.TinyTorchTest(model, _loss, optim, batch, supervised=False)

    test.test(
        train_vars=list(model.named_parameters()),
        test_vars_change=True,
        test_inf_vals=True,
        test_nan_vals=True,
    )

def test_multiarg_unsupervised():
    """Tests a multiple argument unsupervised regression problem"""

    # Sets random seed
    torch.manual_seed(1)

    # Model
    layers = [3, 10, 1]
    model = test_networks.MultiArgRegression(layers)

    # Data
    batch = [torch.rand(4, 2), torch.rand(4,1)]

    # Optimiser
    optim = torch.optim.Adam([p for p in model.parameters() if p.requires_grad])

    # Loss
    def _loss(output):
        return torch.mean(output**2)

    # Setup test suite
    test = ttt.TinyTorchTest(model, _loss, optim, batch, supervised=False)

    test.test(
        train_vars=list(model.named_parameters()),
        test_vars_change=True,
        test_inf_vals=True,
        test_nan_vals=True,
    )


def test_classification():
    """Tests a classification network"""

    # Sets random seed
    torch.manual_seed(1)

    # Model
    layers = [3, 10, 1]
    model = test_networks.SingleArgClassification(layers)

    # Data
    # the multiplying by 100 is used to force the use of the whole output range (from 0 to 1).
    batch = [torch.rand(4, 3) * 100, torch.zeros(4, 1)]

    # Optimiser
    optim = torch.optim.Adam([p for p in model.parameters() if p.requires_grad])

    # Loss
    loss_fn = torch.nn.CrossEntropyLoss()

    # Setup test suite
    test = ttt.TinyTorchTest(model, loss_fn, optim, batch)

    test.test(test_output_range=True)

    # Checks below range exception
    with pytest.raises(ttt.RangeException):
        test.test(output_range=(0.5,1), test_output_range=True)

    # Checks above range exception
    with pytest.raises(ttt.RangeException):
        test.test(output_range=(0, 0.5), test_output_range=True)


def test_params_dont_change():
    """Tests if parameters don't train"""

    # Sets seed
    torch.manual_seed(1)

    # Sets up the model
    inputs = torch.rand(20,20)
    targets = torch.rand(20,2)
    batch = [inputs, targets]
    model = torch.nn.Linear(20,2)

    # Avoids training the bias term
    params_to_train = [ param[1] for param in model.named_parameters() if param[0] != 'bias']

    # Setup test suite
    test = ttt.TinyTorchTest(
        model,
        torch.nn.functional.cross_entropy,
        torch.optim.Adam(params_to_train),
        batch,
    )

    # Checks the bias term changes
    test.test(
        non_train_vars=[('bias', model.bias)],
    )

    # Checks an error is raised when checking that all variables change
    with pytest.raises(ttt.VariablesChangeException):
        test.test(
            train_vars=[('bias', model.bias)],
        )

    # Checks an error is  raised when the bias term changes
    with pytest.raises(ttt.VariablesChangeException):
        test.test(
            non_train_vars=list(model.named_parameters()),
        )


def test_nan_exception():
    """Tests for NaN exception"""

    # Sets seed
    torch.manual_seed(1)

    # Model
    layers = [3, 10, 1]
    model = test_networks.SingleArgRegression(layers)

    # Data
    batch = [torch.rand(4, 3), torch.rand(4,1)]

    batch[0][0, 0] = float('nan')

    # Optimiser
    optim = torch.optim.Adam([p for p in model.parameters() if p.requires_grad])

    # Loss
    loss_fn = torch.nn.MSELoss()

    # Setup test suite
    test = ttt.TinyTorchTest(
        model,
        loss_fn,
        optim,
        batch,
    )

    # run all tests
    with pytest.raises(ttt.NaNTensorException):
        test.test(test_nan_vals=True,)

def test_inf_exception():
    """Tests for inf exception"""

    # Sets seed
    torch.manual_seed(1)

    # Model
    layers = [3, 10, 1]
    model = test_networks.InfModel(layers)

    # Data
    batch = [torch.rand(4, 3), torch.rand(4,1)]

    # Optimiser
    optim = torch.optim.Adam([p for p in model.parameters() if p.requires_grad])

    # Loss
    loss_fn = torch.nn.MSELoss()

    # Setup test suite
    test = ttt.TinyTorchTest(
        model,
        loss_fn,
        optim,
        batch,
    )

    # run all tests
    with pytest.raises(ttt.InfTensorException):
        test.test(test_inf_vals=True)

def test_gpu():
    """Tests for GPU exception"""

    # Model
    layers = [3, 10, 1]
    model = test_networks.SingleArgRegression(layers)

    # Data
    batch = [torch.rand(4, 3), torch.rand(4,1)]

    # Optimiser
    optim = torch.optim.Adam([p for p in model.parameters() if p.requires_grad])

    # Loss
    loss_fn = torch.nn.MSELoss()

    # Sets up test suite
    test = ttt.TinyTorchTest(
        model,
        loss_fn,
        optim,
        batch,
    )

    # Checks if the GPU is available
    # Alternatively, checks an exception is raised
    if torch.cuda.is_available():
        test.test(test_gpu_available=True)

    else:
        with pytest.raises(ttt.GpuUnusedException):
            test.test(test_gpu_available=True)

if __name__ == '__main__':
    print("Running tests...")
    test_regression()
    test_regression_multi_args()
    test_regression_unsupervised()
    test_multiarg_unsupervised()
    test_classification()
    test_params_dont_change()
    test_nan_exception()
    test_inf_exception()
    test_gpu()
    print("Testing complete!")
