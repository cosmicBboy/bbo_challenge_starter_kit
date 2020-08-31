import traceback
from copy import deepcopy
from typing import Dict, List, NamedTuple, Any, Union

import numpy as np
import scipy.stats as ss
import torch
import torch.nn as nn

from agent import Agent, Memory
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
        n_per_agent=20,
        hidden_size=32,
        dropout=0.01,
        learning_rate=0.03,
        weight_decay=0.1,
        entropy_coef=0.01,
        clip_grad=1.0,
        success_threshold=0,
        failure_tolerance=100,
    ):
        """Build wrapper class to use optimizer in benchmark.

        Parameters
        ----------
        api_config : dict-like of dict-like
            Configuration of the optimization variables. See API description.
        """
        AbstractOptimizer.__init__(self, api_config)

        self.n_per_agent = n_per_agent
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
            success_threshold=success_threshold,
            failure_tolerance=failure_tolerance,
        )

        # TODO:
        # 1 Agent Case:
        # - first suggestion: select random set of points
        # - > 1 suggestion: use best point from first suggestion as anchor
        #   point for agent
        # - update anchor points based on best suggestion so far
        #
        # N agent case:
        # - first suggestion: select random set of points
        # - > 1 suggestion: use best point from first suggestion as anchor
        #   points for all agents
        # - experiment:
        #   - variant 1: independently update anchor points based on best
        #     suggestion found per agent
        #   - variant 2: globally update anchor points based on best suggestion
        #     found across all agents
        #
        # Experiment
        # - scale covariance matrix to constrain agent actions to some percent
        #   of the domain [0, 1]
        # - make scaling value a learnable parameter?

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
            suggestions = torch.stack(
                self.swarm(
                    n_suggestions,
                    n_suggestions
                    if self.n_per_agent is None
                    else self.n_per_agent,
                )
            )
            # make sure suggestions are within bounds
            suggestions = from_unit_cube(
                suggestions.detach().numpy(),
                self.lb, self.ub
            )
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
            loss.backward()

            grad_norm = 0.0
            print(
                "[selected agents]",
                self.swarm.history.collective_memories[-1].agent_index
            )
            # TODO: should clip gradients for all agents since they are also
            # evaluating the value of actions off-policy
            for local_agent in self.swarm.local_agents:
                nn.utils.clip_grad_norm_(
                    local_agent.agent.parameters(), self.clip_grad
                )
                grad_norm += compute_grad_norm(local_agent.agent)
            grad_norm /= self.swarm.n_agents

            self.swarm.optim.step()

            best_y = min(y)
            print(
                {
                    "mean_y": f"{np.mean(y):0.04f}",
                    "best_y": f"{best_y:0.04f}",
                    "loss": f"{loss.detach().item():0.04f}",
                    "grad_norm": f"{grad_norm:0.04f}",
                },
                f"\n{'-' * 30}",
            )
        except Exception:
            traceback.print_exc()


if __name__ == "__main__":
    experiment_main(SwarmOptimizer)
