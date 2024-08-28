import torch
from torch import nn

# TODO: Allow selecting terms of given degrees.

# TODO: Check dimensions of the input and output tensors.


class TaylorPolynomial(nn.Module):
    def __init__(self, num_inputs, num_outputs, degree):
        super().__init__()
        self.expansion_point = nn.Parameter(torch.randn(num_inputs))
        self.derivatives = []
        for k in range(degree + 1):
            for derivative_orders in generate_tuples(num_inputs, k):
                self.derivatives.append(
                    (
                        torch.tensor(derivative_orders),
                        nn.Parameter(torch.randn(num_outputs)),
                    )
                )

    def forward(self, x):
        delta_x = x - self.expansion_point
        terms = []
        for derivative_orders, derivative in self.derivatives:
            powers = torch.pow(delta_x, derivative_orders).prod()
            factorials = factorial(derivative_orders).prod()
            terms.append(derivative * powers / factorials)
        return torch.stack(terms).sum()


def generate_tuples(n, k):
    """Generate all tuples of length n with elements in [0, k] that sum to k."""
    # Base case: when n is 1, there's only one tuple (k,).
    if n == 1:
        yield (k,)
    else:
        # Iterate over all possible first elements of the tuple.
        for i in range(k + 1):
            # Recursively generate the remaining elements.
            for rest in generate_tuples(n - 1, k - i):
                yield (i,) + rest


def log_factorial(x):
    return torch.lgamma(x + 1)


def factorial(x):
    return torch.exp(log_factorial(x))
