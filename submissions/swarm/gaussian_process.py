import math

import torch
import gpytorch
from gpytorch.constraints.constraints import Interval


class CriticGP(gpytorch.models.ExactGP):
    def __init__(
        self,
        X,
        y,
        likelihood,
        ndims,
        lengthscale_constraint,
        outputscale_constraint,
    ):
        super(CriticGP, self).__init__(X, y, likelihood)
        self.ard_dims = ndims
        self.mean_module = gpytorch.means.ConstantMean()
        base_kernel = gpytorch.kernels.MaternKernel(
            lengthscale_constraint=lengthscale_constraint, ard_num_dims=ndims,
        )
        self.covar_module = gpytorch.kernels.ScaleKernel(
            base_kernel, outputscale_constraint=outputscale_constraint,
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def train_gp_model(
    X, y, num_steps, hypers=None,
):
    noise_constraint = Interval(5e-4, 0.2)
    # lengthscale_constraint = Interval(0.005, math.sqrt(X.shape[1]))
    lengthscale_constraint = Interval(0.005, 2.0)
    outputscale_constraint = Interval(0.05, 20.0)

    likelihood = gpytorch.likelihoods.GaussianLikelihood(
        noise_constraint=noise_constraint
    )
    model = CriticGP(
        X,
        y,
        likelihood,
        ndims=X.shape[1],
        lengthscale_constraint=lengthscale_constraint,
        outputscale_constraint=outputscale_constraint,
    )

    model.train()
    likelihood.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    if hypers is not None:
        model.load_state_dict(hypers)
    else:
        model.initialize(
            **{
                "covar_module.outputscale": 1.0,
                "covar_module.base_kernel.lengthscale": 0.5,
                "likelihood.noise": 0.005,
            }
        )

    for i in range(num_steps):
        optimizer.zero_grad()
        output = model(X)
        loss = -mll(output, y)
        loss.backward()
        optimizer.step()

    model.eval()
    likelihood.eval()
    return model
