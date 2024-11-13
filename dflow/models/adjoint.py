import torch


# Define time-dependent functions a, b, f, u, and sigma as example placeholders
def a(X_t, u, t):
    # Example function for a, dependent on X_t and u
    return X_t @ u.t()  # Adjust based on your specific function form


def b(X_t, t):
    # Example function for b, dependent on X_t and t
    return X_t ** 2 + t


def f(X_t, t):
    # Example function for f, dependent on X_t and t
    return X_t.sum(dim=1)


def u(X_t, t):
    # Example control input u, dependent on X_t and t
    return X_t * torch.sin(t)


def sigma(t):
    # Example function for sigma, dependent only on t
    return torch.sin(t)


# Automatic approach using autograd.functional.vjp
def vector_jacobian_product_vjp(X_t, u, t):
    # Compute a(t; X, u)
    a_val = a(X_t, u, t)

    # Define b_term to be differentiable for automatic VJP
    b_term = b(X_t, t) + sigma(t) * u(X_t, t)
    f_term = f(X_t, t) + 0.5 * (u(X_t, t).norm() ** 2)

    # Gradient term for f
    grad_f = torch.autograd.grad(f_term.sum(), X_t, create_graph=True)[0]

    # Vector Jacobian Product of b_term
    vjp_func = torch.func.vjp(b_term, X_t)[1]
    vjp_result_func = vjp_func(a_val)
    print("VJP result with torch.func.vjp:", vjp_result_func)

    # Using torch.autograd.functional.vjp
    vjp_term = torch.autograd.functional.vjp(b_term, X_t, v=a_val)[1]   # torch.func.vjp (batch) vs. torch.autograd.functional.vjp (not batch)
    print("VJP result with torch.autograd.vjp:", vjp_term)
    return -(vjp_term + grad_f)



import torch
from torch import nn
from torchdiffeq import odeint
# Instantiate the neural network

e_model = DerivativeNN()

# Known initial condition at t = 1
a_1 = torch.tensor([[2.0]], requires_grad=True)  # Shape (batch, 1) for torchdiffeq compatibility
num_steps = 100
t_span = torch.linspace(1.0, 0.0, num_steps)


# Integrate the differential equation from t=1 to t=0
a_0 = odeint(e_model, a_1, t_span, method='dopri5')[-1]   # TODO: you can direcly use for-loop to achieve a_0 from a_1 as we know the backward ODE
print("Value of a at time 0:", a_0)
