# Tiny Torchtest

![coverage](.coverage.svg)

A Tiny Test Suite for pytorch based Machine Learning models, inspired by
[mltest](https://github.com/Thenerdstation/mltest/blob/master/mltest/mltest.py).
Chase Roberts lists out 4 basic tests in his [medium
post](https://medium.com/@keeper6928/mltest-automatically-test-neural-network-models-in-one-function-call-eb6f1fa5019d)
about mltest. tinytorchtest is mostly a pytorch port of mltest (which was
written for tensorflow).

--- 

Forked from [BrainPugh](https://github.com/BrianPugh/torchtest) who
forked the repo from
[suriyadeepan](https://github.com/suriyadeepan/torchtest).

Notable changes:

-   Support for models to have multiple positional arguments.

-   Support for unsupervised learning.

- 	Object orientated implementation.

- 	Easily reproducible tests - thanks to the object orientated implementation!

-   Fewer requirements (due to streamlining testing).

-   More comprehensive internal unit tests.

-   This repository is still active. I've created an
    [issue](https://github.com/suriyadeepan/torchtest/issues/6) to
    double check but it looks like the original maintainer is no longer
    actioning pull requests.

---

# Installation

``` bash
pip install --upgrade tinytorchtest
```

# Usage

``` python
# imports for examples
import torch
import torch.nn as nn
```

## Variables Change

``` python
from tinytorchtest import tinytorchtest as ttt

# We'll be using a simple linear model
model = nn.Linear(20, 2)

# For this example, we'll pretend we have a classification problem
# and create some random inputs and outputs.
inputs = torch.randn(20, 20)
targets = torch.randint(0, 2, (20,)).long()
batch = [inputs, targets]

# Next we'll need a loss function
loss_fn = nn.functional.cross_entropy()

# ... and an optimisation function
optim = torch.optim.Adam(model.parameters())

# Lets set up the test object
test = ttt.TinyTorchTest(model, loss_fn, optim, batch)

# Now we've got our tiny test object, lets run some tests!
# What are the variables?
print('Our list of parameters', [ np[0] for np in model.named_parameters() ])

# Do they change after a training step?
#  Let's run a train step and see
test.test(test_vars_change=True)
```

``` python
""" FAILURE """
# Let's try to break this, so the test fails
params_to_train = [ np[1] for np in model.named_parameters() if np[0] is not 'bias' ]
# Run test now
test.test(test_vars_change=True)
# YES! bias did not change
```

## Variables Don't Change

``` python
# What if bias is not supposed to change, by design?
#  Let's test to see if bias remains the same after training
test.test(non_train_vars=[('bias', model.bias)])
# It does! Good. Now, let's move on.
```

## Output Range

``` python
# NOTE : bias is fixed (not trainable)
test.test(output_range=(-2, 2), test_output_range=True)

# Seems to work...
```

``` python
""" FAILURE """
#  Let's tweak the model to fail the test.
model.bias = nn.Parameter(2 + torch.randn(2, ))

# We'll still use the same loss function, optimiser and batch
# from earlier; however this time we've tweaked the bias of the model.
# As it's a new model, we'll need a new tiny test object.
test = ttt.TinyTorchTest(model , loss_fn, optim, batch)

test.test(output_range=(-1, 1), test_output_range=True)

# As expected, it fails; yay!
```

## NaN Tensors

``` python
""" FAILURE """

# Again, keeping everything the same but tweaking the model
model.bias = nn.Parameter(float('NaN') * torch.randn(2, ))

test = ttt.TinyTorchTest(model , loss_fn, optim, batch)

test.test(test_nan_vals=True)
# This test should fail as we've got 'NaN' values in the outputs.
```

## Inf Tensors

``` python
""" FAILURE """
model.bias = nn.Parameter(float('Inf') * torch.randn(2, ))

test = ttt.TinyTorchTest(model , loss_fn, optim, batch)

test.test(test_inf_vals=True)
# Again, this will fail as we've now got 'Inf' values in our model outputs.
```

## Multi-argument models
``` python
# Everything we've done works for models with multi-arguments

# Let's define a network that takes some input features along 
# with a 3D spacial coordinate and predicts a single value.
# Sure, we could perform the concatenation before we pass 
# our inputs to the model but let's say that it's much easier to
# do it this way. Maybe as you're working tightly with other codes
# and you want to match your inputs with the other code.
class MutliArgModel(torch.nn.Module):
	def __init__(self):
		self.layers = torch.nn.Linear(8, 1)
	def foward(self, data, x, y, z):
		inputs = torch.cat((data, x, y, z), dim=1)
		return self.layers(nn_input)
model = MultiArgModel()

# This looks a bit more like a regression problem so we'll redefine our loss 
# function to be something more appropriate.
loss_fn = torch.nn.MSELoss()

# We'll stick with the Adam optimiser but for completeness lets redefine it below
optim = Adam(model.parameters())

# We'll also need some new data for this model
inputs = (
	torch.rand(10, 5), # data
	torch.rand(10, 1), # x
	torch.rand(10, 1), # y
	torch.rand(10, 1), # z
)
outputs = torch.rand(10,1)
batch = [inputs, outputs]
		
# Next we initialise our tiny test object
test = ttt.TinyTorchTest(model , loss_fn, optim, batch)

# Now lets run some tests
test.test(
	train_vars=list(model.named_parameters()),
	test_vars_change=True,
	test_inf_vals=True,
	test_nan_vals=True,
)
# Great! Everything works as before but with models that take multiple inputs.
```

## Models with tuple or list outputs

``` python
# Now what about models that output a tuple or list of tensors?
# This could be for something like a variation auto-encoder
# or anything where it's more convienient to separate 
# internal networks.

# Lets define a model
class MultiOutputModel(nn.Module):
	def __init__(self, in_size, hidden_size, out_size, num_outputs):
		super().__init__()

		# This network is common for all predictions.
		nets = [nn.Linear(in_size, hidden_size)] 

		# These networks operate separately (in parallel)
		for _ in range(num_outputs):
			nets.append(nn.Linear(hidden_size, out_size))
		self.nets = nn.ModuleList(nets)

	def forward(self, x):
		# Passes through the first network
		x = self.nets[0](x)

		# Returns a list of the seprate network predictions
		return [net(x) for net in self.nets[1:]]

# 10 features, 5 hidden nodes, 1 output node, 3 output models
model = MultiOutputModel(10, 5, 1, 3)

# Creates a batch with 100 samples.
batch = [torch.rand(100, 10), torch.rand(100, 1)]

# Optimiser...
optim = torch.optim.Adam([p for p in model.parameters() if p.requires_grad])

# Here we'll have to define a custom loss function to deal with the multiple outputs
# For now, I'll use something trivial (and quite meaningless).
def _loss(outputs, target):
	loss_list = [torch.abs(output - target) ** p for p, output in enumerate(outputs)]
	return torch.mean(torch.tensor(loss_list))

# Setup test suite
test = ttt.TinyTorchTest(model, _loss, optim, batch, supervised=True)

# Run the tests we want to run!
test.test(
	train_vars=list(model.named_parameters()),
	test_vars_change=True,
	test_inf_vals=True,
	test_nan_vals=True,
)

# Great! Everything works as before but with model with tuple or list outputs.
```

## Unsupervised learning

``` python
# We've looked a lot at supervised learning examples
# but what about unsupervised learning?

# Lets define a simple model
model = nn.Linear(20, 2)

# Now our inputs - notice there are no labels so we just have inputs in our batch
batch = torch.randn(20, 20)

# Here we'll write a very basic loss function that represents a reconstruction loss.
# This is actually a mean absolute distance loss function.
# This would typically be used for something like an auto-encoder.
# The important thing to note is tinytorchtest expects the loss to be loss(outputs, inputs).
def loss_fn(output, input):
	return torch.mean(torch.abs(output - input))

# We set supervised to false, to let the test suite
# know that there aren't any targets or correct labels.
test = ttt.TinyTorchTest(model , loss_fn, optim, batch, supervised=False)

# Now lets run some tests
test.test(
	train_vars=list(model.named_parameters()),
	test_vars_change=True,
	test_inf_vals=True,
	test_nan_vals=True,
)
# Great! Everything works as before but with unsupervised models.
```

## Testing the GPU

``` python
# Some models really need GPU availability.
# We can get our test suite to fail when the GPU isn't available.

# Sticking with the unsupervised example
test = ttt.TinyTorchTest(model , loss_fn, optim, batch, supervised=False)

# Now lets make sure the GPU is available.
test.test(test_gpu_available=True)
# This test will fail if the GPU isn't available. Your CPU can thank you later.

# We can also explitly ask that our model and tensors be moved to the GPU
test = ttt.TinyTorchTest(model , loss_fn, optim, batch, supervised=False, device='cuda:0')

# Now all future tests will be run on the GPU
```

## Reproducible tests

``` python
# When unit testing our models it's good practice to have reproducable results.
# For this, we can spefiy a seed when getting our tiny test object.
test = ttt.TinyTorchTest(model, loss_fn, optim, batch, seed=42)

# This seed will be called before running each test so the results should always be the same
# regardless of the order they are called.

```

# Debugging

``` bash
torchtest\torchtest.py", line 151, in _var_change_helper
assert not torch.equal(p0, p1)
RuntimeError: Expected object of backend CPU but got backend CUDA for argument #2 'other'
```

When you are making use of a GPU, you should explicitly specify
`device=cuda:0`. By default `device` is set to `cpu`. See [issue
#1](https://github.com/suriyadeepan/torchtest/issues/1) for more
information.

``` python
test = ttt.TinyTorchTest(model , loss_fn, optim, batch, device='cuda:0')
```

# Citation

``` tex
@misc{abdrysdale2022
  author = {Alex Drysdale},
  title = {tinytorchtest},
  year = {2022},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/abdrysdale/tinytorchtest}},
  commit = {4c39c52f27aad1fe9bcc7fbb2525fe1292db81b7}
 }
@misc{Ram2019,
  author = {Suriyadeepan Ramamoorthy},
  title = {torchtest},
  year = {2019},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/suriyadeepan/torchtest}},
  commit = {42ba442e54e5117de80f761a796fba3589f9b223}
}
```
