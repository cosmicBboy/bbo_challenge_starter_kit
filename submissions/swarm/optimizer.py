import traceback
from typing import Dict, List, NamedTuple, Any, Union

import numpy as np
import scipy.stats as ss
import torch
import torch.nn as nn

from agent import Agent
from swarm import Swarm
from utils import (
    copula_standardize,
    derive_reward,
    compute_grad_norm,
    latin_hypercube,
    to_unit_cube,
    from_unit_cube,
)

from bayesmark.abstract_optimizer import AbstractOptimizer
from bayesmark.experiment import experiment_main
from bayesmark.space import JointSpace


Array = Union[torch.Tensor, np.ndarray]
Hyperparameter = Union[int, float, str]


class SwarmOptimizer(AbstractOptimizer):
    primary_import = None

    def __init__(
        self,
        api_config,
        n_agents=1,
        n_candidates=20,
        n_iter=200,
        num_gp_update_steps=200,
        n_perturb=3,
        noise_window=0.1,
        hidden_size=64,
        dropout=0.1,
        learning_rate=0.3,
        weight_decay=1000.0,
        entropy_coef=0.1,
        clip_grad=1.0,
    ):
        """Build wrapper class to use optimizer in benchmark.

        Parameters
        ----------
        api_config : dict-like of dict-like
            Configuration of the optimization variables. See API description.
        """
        AbstractOptimizer.__init__(self, api_config)

        self.n_candidates = n_candidates
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.entropy_coef = entropy_coef
        self.clip_grad = clip_grad

        self.space_x = JointSpace(api_config)
        self.bounds = self.space_x.get_bounds()
        self.lb, self.ub = self.bounds[:, 0], self.bounds[:, 1]
        self.dim = len(self.bounds)

        # instantiate agents
        self.n_actions = len(api_config)
        self.swarm = Swarm(
            n_agents=n_agents,
            agent_config={
                "n_actions": self.n_actions,
                "hidden_size": hidden_size,
                "dropout": dropout,
            },
            anchor_fn=lambda: latin_hypercube(self.n_actions, self.dim)[0],
            optim=torch.optim.Adam,
            optim_config={
                "lr": learning_rate,
                "weight_decay": self.weight_decay,
            },
            num_gp_update_steps=num_gp_update_steps,
        )

    def suggest(self, n_suggestions=1):
        """Get suggestions from the optimizer.

        Parameters
        ----------
        n_suggestions : int
            Desired number of parallel suggestions in the output

        Returns
        -------
        next_guess : list of dict
            List of `n_suggestions` suggestions to evaluate the objective
            function. Each suggestion is a dictionary where each key
            corresponds to a parameter being optimized.
        """
        try:
            suggestions = self.swarm(
                n_suggestions,
                self.n_candidates,
                self.n_iter,
            )
            # make sure suggestions are within bounds
            suggestions = from_unit_cube(
                suggestions.detach().numpy(), self.lb, self.ub
            )
            print("[suggestions]\n", suggestions)
            suggestions = self.space_x.unwarp(suggestions)
            return suggestions
        except Exception:
            traceback.print_exc()

    def observe(self, X, y):
        """Feed an observation back.

        Parameters
        ----------
        X : list of dict-like
            Places where the objective function has already been evaluated.
            Each suggestion is a dictionary where each key corresponds to a
            parameter being optimized.
        y : array-like, shape (n,)
            Corresponding values where objective has been evaluated
        """
        try:
            self.swarm.optim.zero_grad()
            loss = self.swarm.update(y, self.entropy_coef)
            if loss is None:
                loss, grad_norm = np.nan, np.nan
            else:
                loss.backward()

                grad_norm = 0.0
                print(
                    "[selected agents]",
                    self.swarm.history.collective_memories[-1].agent_ids,
                )
                for local_agent in self.swarm.local_agents:
                    nn.utils.clip_grad_norm_(
                        local_agent.agent.parameters(), self.clip_grad
                    )
                    grad_norm += compute_grad_norm(local_agent.agent)
                grad_norm /= self.swarm.n_agents
                loss = loss.detach().item()
                self.swarm.optim.step()

            print(
                {
                    "mean_y": f"{np.mean(y):0.04f}",
                    "best_y": f"{min(y):0.04f}",
                    "loss": f"{loss:0.04f}",
                    "grad_norm": f"{grad_norm:0.04f}",
                },
                f"\n{'-' * 30}",
            )
        except Exception:
            traceback.print_exc()


if __name__ == "__main__":
    experiment_main(SwarmOptimizer)
