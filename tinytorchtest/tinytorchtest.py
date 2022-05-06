"""torchtest : A Tiny Test Suite for PyTorch

A tiny test suite for pytorch based Machine Learning models, inspired by mltest.

Chase Roberts lists out 4 basic tests in his medium post about mltest.
https://medium.com/@keeper6928/mltest-automatically-test-neural-network-models-in-one-function-call-eb6f1fa5019d

torchtest is sort of a pytorch port of mltest (which was written for tensorflow models).
"""

import torch

# default model output range
MODEL_OUT_LOW = -1
MODEL_OUT_HIGH = 1

class GpuUnusedException(Exception): # pylint: disable=missing-class-docstring
    pass

class VariablesChangeException(Exception): # pylint: disable=missing-class-docstring
    pass

class RangeException(Exception): # pylint: disable=missing-class-docstring
    pass

class DependencyException(Exception): # pylint: disable=missing-class-docstring
    pass

class NaNTensorException(Exception): # pylint: disable=missing-class-docstring
    pass

class InfTensorException(Exception): # pylint: disable=missing-class-docstring
    pass

def setup(seed=0):
    """Set random seed for torch"""
    torch.manual_seed(seed)

def _pack_batch(tensor_or_tuple, device):
    """ Packages object ``tensor_or_tuple`` into a tuple to be unpacked.

    Recursively transfers all tensor objects to device

    Parameters
    ----------
    tensor_or_tuple : torch.Tensor or tuple containing torch.Tensor
    device : str

    Returns
    -------
    tuple
            positional arguments
    """

    def _helper(tensor_or_tuple):
        if isinstance(tensor_or_tuple, torch.Tensor):
            tensor_or_tuple.to(device)
            return tensor_or_tuple

        output = [_helper(item) for item in tensor_or_tuple]
        return output


    if isinstance(tensor_or_tuple, torch.Tensor):
        # For backwards compatability
        return (tensor_or_tuple,)
    return _helper(tensor_or_tuple)


def _train_step(model, loss_fn, optim, batch, device, supervised=True):
    """Run a training step on model for a given batch of data

    Parameters of the model accumulate gradients and the optimizer performs
    a gradient update on the parameters

    Parameters
    ----------
    model : torch.nn.Module
        torch model, an instance of torch.nn.Module
    loss_fn : function
        a loss function from torch.nn.functional
    optim : torch.optim.Optimizer
        an optimizer instance
    batch : list
        a 2 element list of inputs and labels, to be fed to the model
    supervised : bool, optional
        If true, expects batch to contain [inputs, targets].
        Else, expects batch to be the model inputs.
        If supervised=False then the loss_fn is only fed in the model outputs.
        Defaults to True.
    """

    # put model in train mode
    model.train()
    model.to(device)

    #  run one forward + backward step
    # clear gradient
    optim.zero_grad()

    # inputs and targets
    if supervised:
        inputs, targets = batch[0], batch[1]    # Need to recursively move these to device
        targets = _pack_batch(targets, device) # Moves targets to device

    else:
        inputs = batch

    # Moves inputs to device
    inputs = _pack_batch(inputs, device)

    # forward
    outputs = model(*inputs)

    # Gets loss
    if supervised:
        loss = loss_fn(outputs, *targets)
    else:
        loss = loss_fn(outputs)

    # backward
    loss.backward()
    # optimization step
    optim.step()

def _forward_step(model, batch, device):
    """Run a forward step of model for a given batch of data

    Parameters
    ----------
    model : torch.nn.Module
        torch model, an instance of torch.nn.Module
    batch : list
        a 2 element list of inputs and labels, to be fed to the model

    Returns
    -------
    torch.tensor
        output of model's forward function
    """

    # put model in eval mode
    model.eval()
    model.to(device)

    with torch.no_grad():
        # inputs and targets
        inputs = batch[0]
        # move data to DEVICE
        inputs = _pack_batch(inputs, device)
        # forward
        return model(*inputs)

def _var_change_helper(vars_change, model, loss_fn, optim, batch, device, params=None, **kwargs):
    """Check if given variables (params) change or not during training

    If parameters (params) aren't provided, check all parameters.

    Parameters
    ----------
    vars_change : bool
        a flag which controls the check for change or not change
    model : torch.nn.Module
        torch model, an instance of torch.nn.Module
    loss_fn : function
        a loss function from torch.nn.functional
    optim : torch.optim.Optimizer
        an optimizer instance
    batch : list
        a 2 element list of inputs and labels, to be fed to the model
    params : list, optional
        list of parameters of form (name, variable)
    **kwarg supervised : bool
        True for supervised learning models. False otherwise.

    Raises
    ------
    VariablesChangeException
        if vars_change is True and params DO NOT change during training
        if vars_change is False and params DO change during training
    """

    if params is None:
        # get a list of params that are allowed to change
        params = [ np for np in model.named_parameters() if np[1].requires_grad ]

    # take a copy
    initial_params = [ (name, p.clone()) for (name, p) in params ]

    # run a training step
    _train_step(model, loss_fn, optim, batch, device, **kwargs)

    # check if variables have changed
    for (_, param_0), (name, param_1) in zip(initial_params, params):
        try:
            if vars_change:
                assert not torch.equal(param_0.to(device), param_1.to(device))
            else:
                assert torch.equal(param_0.to(device), param_1.to(device))
        except AssertionError as error:
            msg = 'did not change!' if vars_change else 'changed!'
            raise VariablesChangeException(f"{name} {msg}") from error

def assert_uses_gpu():
    """Make sure GPU is available and accessible

    Raises
    ------
    GpuUnusedException
        If GPU is inaccessible
    """

    try:
        assert torch.cuda.is_available()
    except AssertionError as error:
        raise GpuUnusedException("GPU inaccessible") from error

def assert_vars_change(model, loss_fn, optim, batch, device, params=None, **kwargs):
    """Make sure that the given parameters (params) DO change during training

    If parameters (params) aren't provided, check all parameters.

    Parameters
    ----------
    model : torch.nn.Module
        torch model, an instance of torch.nn.Module
    loss_fn : function
        a loss function from torch.nn.functional
    optim : torch.optim.Optimizer
        an optimizer instance
    batch : list
        a 2 element list of inputs and labels, to be fed to the model
    params : list, optional
        list of parameters of form (name, variable)
    **kwarg supervised : bool
        True for supervised learning models. False otherwise.

    Raises
    ------
    VariablesChangeException
        If params do not change during training
    """

    _var_change_helper(True, model, loss_fn, optim, batch, device, params, **kwargs)

def assert_vars_same(model, loss_fn, optim, batch, device, params=None, **kwargs):
    """Make sure that the given parameters (params) DO NOT change during training

    If parameters (params) aren't provided, check all parameters.

    Parameters
    ----------
    model : torch.nn.Module
        torch model, an instance of torch.nn.Module
    loss_fn : function
        a loss function from torch.nn.functional
    optim : torch.optim.Optimizer
        an optimizer instance
    batch : list
        a 2 element list of inputs and labels, to be fed to the model
    params : list, optional
        list of parameters of form (name, variable)
    **kwarg supervised : bool
        True for supervised learning models. False otherwise.

    Raises
    ------
    VariablesChangeException
        If params change during training
    """

    _var_change_helper(False, model, loss_fn, optim, batch, device, params, **kwargs)

def assert_any_greater_than(tensor, value):
    """Make sure that one or more elements of tensor greater than value

    Parameters
    ----------
    tensor : torch.tensor
        input tensor
    value : float
        numerical value to check against

    Raises
    ------
    RangeException
        If all elements of tensor are less than value
    """

    try:
        assert (tensor > value).byte().any()
    except AssertionError as error:
        raise RangeException(f"All elements of tensor are less than {value}") from error

def assert_all_greater_than(tensor, value):
    """Make sure that all elements of tensor are greater than value

    Parameters
    ----------
    tensor : torch.tensor
        input tensor
    value : float
        numerical value to check against

    Raises
    ------
    RangeException
        If one or more elements of tensor are less than value
    """

    try:
        assert (tensor > value).byte().all()
    except AssertionError as error:
        raise RangeException(f"Some elements of tensor are less than {value}") from error

def assert_any_less_than(tensor, value):
    """Make sure that one or more elements of tensor are less than value

    Parameters
    ----------
    tensor : torch.tensor
        input tensor
    value : float
        numerical value to check against

    Raises
    ------
    RangeException
        If all elements of tensor are greater than value
    """

    try:
        assert (tensor < value).byte().any()
    except AssertionError as error:
        raise RangeException(f"All elements of tensor are greater than {value}") from error

def assert_all_less_than(tensor, value):
    """Make sure that all elements of tensor are less than value

    Parameters
    ----------
    tensor : torch.tensor
        input tensor
    value : float
        numerical value to check against

    Raises
    ------
    RangeException
        If one or more elements of tensor are greater than value
    """

    try:
        assert (tensor < value).byte().all()
    except AssertionError as error:
        raise RangeException(f"Some elements of tensor are greater than {value}") from error

def assert_input_dependency(model, loss_fn, optim, batch,
        independent_vars=None,
        dependent_vars=None):
    """Makes sure the "dependent_vars" are dependent on "independent_vars" """
    raise NotImplementedError("""
        I don't know a clean way to do this
        Doesn't assert_vars_change() cover this?
    """
    )


def assert_never_nan(tensor):
    """Make sure there are no NaN values in the given tensor.

    Parameters
    ----------
    tensor : torch.tensor
        input tensor

    Raises
    ------
    NaNTensorException
        If one or more NaN values occur in the given tensor
    """

    try:
        assert not torch.isnan(tensor).byte().any()
    except AssertionError as error:
        raise NaNTensorException("There was a NaN value in tensor") from error

def assert_never_inf(tensor):
    """Make sure there are no Inf values in the given tensor.

    Parameters
    ----------
    tensor : torch.tensor
        input tensor

    Raises
    ------
    InfTensorException
        If one or more Inf values occur in the given tensor
    """

    try:
        assert torch.isfinite(tensor).byte().any()
    except AssertionError as error:
        raise InfTensorException("There was an Inf value in tensor") from error

def test_suite(model, loss_fn, optim, batch,
        output_range=None,
        train_vars=None,
        non_train_vars=None,
        test_output_range=False,
        test_vars_change=False,
        test_nan_vals=False,
        test_inf_vals=False,
        test_gpu_available=False,
        device='cpu',
        **kwargs,
):
    """Test Suite : Runs the tests enabled by the user

    If output_range is None, output of model is tested against (MODEL_OUT_LOW,
    MODEL_OUT_HIGH).

    Parameters
    ----------
    model : torch.nn.Module
        torch model, an instance of torch.nn.Module
    loss_fn : function
        a loss function from torch.nn.functional
    optim : torch.optim.Optimizer
        an optimizer instance
    batch : list
        a 2 element list of inputs and labels, to be fed to the model
    output_range : tuple, optional
        (low, high) tuple to check against the range of logits (default is
        None)
    train_vars : list, optional
        list of parameters of form (name, variable) to check if they change
        during training (default is None)
    non_train_vars : list, optioal
        list of parameters of form (name, variable) to check if they DO NOT
        change during training (default is None)
    test_output_range : boolean, optional
        switch to turn on or off range test (default is False)
    test_vars_change : boolean, optional
        switch to turn on or off variables change test (default is False)
    test_nan_vals : boolean, optional
        switch to turn on or off test for presence of NaN values (default is False)
    test_inf_vals : boolean, optional
        switch to turn on or off test for presence of Inf values (default is False)
    test_gpu_available : boolean, optional
        switch to turn on or off GPU availability test (default is False)
    **kwarg supervised : bool
        True for supervised learning models. False otherwise.

    Raises
    ------
    VariablesChangeException
        If selected params change/do not change during training
    RangeException
        If range of output exceeds the given limit
    GpuUnusedException
        If GPU is inaccessible
    NaNTensorException
        If one or more NaN values occur in model output
    InfTensorException
        If one or more Inf values occur in model output
    """

    # check if all variables change
    if test_vars_change:
        assert_vars_change(model, loss_fn, optim, batch, device, **kwargs)

    # check if train_vars change
    if train_vars is not None:
        assert_vars_change(model, loss_fn, optim, batch, device, params=train_vars, **kwargs)

    # check if non_train_vars don't change
    if non_train_vars is not None:
        assert_vars_same(model, loss_fn, optim, batch, device, params=non_train_vars, **kwargs)

    # run forward once
    model_out = _forward_step(model, batch, device)

    # range tests
    if test_output_range:
        if output_range is None:
            assert_all_greater_than(model_out, MODEL_OUT_LOW)
            assert_all_less_than(model_out, MODEL_OUT_HIGH)
        else:
            assert_all_greater_than(model_out, output_range[0])
            assert_all_less_than(model_out, output_range[1])

    # NaN Test
    if test_nan_vals:
        assert_never_nan(model_out)

    # Inf Test
    if test_inf_vals:
        assert_never_inf(model_out)

    # GPU test
    if test_gpu_available:
        assert_uses_gpu()

    return True
