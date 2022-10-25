"""tinytorchtest : A Tiny Test Suite for PyTorch

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

class NaNTensorException(Exception): # pylint: disable=missing-class-docstring
    pass

class InfTensorException(Exception): # pylint: disable=missing-class-docstring
    pass

class TinyTorchTest():
    """Class for the tiny torch testing suite"""
    def __init__(
        self,
        model,
        loss_fn,
        optim,
        batch,
        device="cpu",
        supervised=True,
        seed=42,
    ):
        """
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
        device : str
            Device to load model and data on to. Defaults to "cpu".
        supervised : bool
            True for supervised learning models. False otherwise.
            Defaults to True.
        seed : int
            Seed for torch manual seed. Will manually set seed before each test.
            Defaults to 42.
        """
        self.model = model
        self.loss_fn = loss_fn
        self.optim = optim
        self.batch = batch
        self.device = device
        self.supervised = supervised
        self.seed = seed

    def _seed(self, seed=None):
        """Sets the seed"""
        if isinstance(seed, type(None)):
            seed = self.seed
        torch.manual_seed(seed)

    def assert_vars_change(self, params=None):
        """Asserts if variables change

        Parameters
        ----------

        params : list, optional
            list of parameters of form (name, variable)
        """
        self._seed()
        return assert_vars_change(
            self.model,
			self.loss_fn,
			self.optim,
			self.batch,
			self.device,
            params=params,
			supervised=self.supervised,
        )

    def assert_vars_same(self, params=None):
        """Asserts if variables don't change

        Parameters
        ----------

        params : list, optional
            list of parameters of form (name, variable)
        """
        self._seed()
        return assert_vars_same(
            self.model,
			self.loss_fn,
			self.optim,
			self.batch,
			self.device,
            params=params,
			supervised=self.supervised,
        )


    def _forward_step(self):
        """Returns one forward step of the model"""
        self._seed()
        return _forward_step(self.model, self.batch, self.device, supervised=self.supervised)


    def test_output_range(self, model_out=None, output_range=(MODEL_OUT_LOW, MODEL_OUT_HIGH)):
        """Checks if the output is within the range

        Parameters
        ----------
        output_range : tuple, optional
            (low, high) tuple to check against the range of logits.
            Defaults to (MODEL_OUT_LOW, MODEL_OUT_HIGH).
        """
        if isinstance(model_out, type(None)):
            model_out = self._forward_step()
        assert_all_greater_than(model_out, output_range[0])
        assert_all_less_than(model_out, output_range[1])


    def test_nan_vals(self, model_out=None):
        """Tests NaN values

        Parameters
        ----------
            model_out : tensor, optional
                If None, gets model output by running forward pass.
                Defaults to None.
        """
        if isinstance(model_out, type(None)):
            model_out = self._forward_step()
        assert_never_nan(model_out)


    def test_inf_vals(self, model_out=None):
        """Tests Inf values

        Parameters
        ----------
            model_out : tensor, optional
                If None, gets model output by running forward pass.
                Defaults to None.
        """
        if isinstance(model_out, type(None)):
            model_out = self._forward_step()
        assert_never_inf(model_out)

    def test_gpu_available(self): # pylint: disable=no-self-use
        """Tests the GPU availability"""
        assert_uses_gpu()


    def test( # pylint: disable=too-many-arguments
        self,
        output_range=(MODEL_OUT_LOW, MODEL_OUT_HIGH),
        train_vars=None,
        non_train_vars=None,
        test_output_range=False,
        test_vars_change=False,
        test_nan_vals=False,
        test_inf_vals=False,
        test_gpu_available=False,
    ):
        """Test Suite : Runs the tests enabled by the user

        If output_range is None, output of model is tested against (MODEL_OUT_LOW,
        MODEL_OUT_HIGH).

        Parameters
        ----------
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

        self._seed()

        # Check if all variables change
        if test_vars_change:
            self.assert_vars_change()

        # Check if train_vars change
        if train_vars is not None:
            self.assert_vars_change(params=train_vars)

        # Check if non_train_vars don't change
        if non_train_vars is not None:
            self.assert_vars_same(params=non_train_vars)

        # Gets an output of the model
        model_out = self._forward_step()

        # Tests output range
        if test_output_range:
            self.test_output_range(model_out=model_out, output_range=output_range)

        # NaN Test
        if test_nan_vals:
            self.test_nan_vals(model_out=model_out)

        # Inf Test
        if test_inf_vals:
            self.test_inf_vals(model_out=model_out)

        # GPU test
        if test_gpu_available:
            self.test_gpu_available()

        return True

def multi_output_support(test):
    """Runs a test on each output in outputs"""
    def _test(outputs, *args, **kwargs):
        if isinstance(outputs, (list, tuple)):
            return [test(output, *args, **kwargs) for output in outputs]
        return test(outputs, *args, **kwargs)
    return _test


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
        loss = loss_fn(outputs, inputs)

    # backward
    loss.backward()
    # optimization step
    optim.step()

def _forward_step(model, batch, device, supervised=True):
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
        if supervised:
            inputs = batch[0]
        else:
            inputs = batch
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

@multi_output_support
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

@multi_output_support
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


@multi_output_support
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

@multi_output_support
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
