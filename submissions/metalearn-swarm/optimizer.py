from copy import deepcopy
from typing import Dict, List, NamedTuple, Any, Union

import numpy as np
import scipy.stats as ss
import torch
import torch.nn as nn

from agent import Agent, Memory

from bayesmark.abstract_optimizer import AbstractOptimizer
from bayesmark.experiment import experiment_main


Array = Union[torch.Tensor, np.ndarray]
Hyperparameter = Union[int, float, str]


class History(NamedTuple):
    memories: List[Memory] = []
    observations: List[Dict[str, Any]] = []
    outcomes: List[List[float]] = []
    rewards: List[List[float]] = []

    def record(self, memory, observations, outcomes, rewards):
        self.memories.append(deepcopy(memory))
        self.observations.append(deepcopy(observations))
        self.outcomes.append(deepcopy(outcomes))
        self.rewards.append(deepcopy(rewards))

    def recall(self, n_actions):
        # get actions, reward, and observation with best outcome
        best_idx = (
            self.rewards[-1].index(max(self.rewards[-1]))
            if self.rewards
            else None
        )

        if best_idx is None:
            prev_actions = np.array([0 for _ in range(n_actions)])
            prev_reward = np.array([0])
        else:
            prev_actions = self.memories[-1].actions[best_idx]
            prev_reward = np.array([self.rewards[-1][best_idx]])

        return (
            torch.from_numpy(prev_actions).float(),
            torch.from_numpy(prev_reward).float(),
        )

    def erase(self):
        del self.memories[:]
        del self.observations[:]
        del self.outcomes[:]
        del self.rewards[:]


def sample_random_actions(api_config):
    random_actions: List[Hyperparameter] = []
    for param, config in api_config.items():
        choice = (
            np.random.randint(*config["range"])
            if config["type"] == "int"
            else np.random.uniform(*config["range"])
            if config["type"] == "real"
            else np.random.choice([0, 1])
            if config["type"] == "bool"
            else np.random.randint(0, len(config["values"]))
            if config["type"] == "cat"
            else None
        )
        if choice is None:
            raise ValueError(f"type {config['type']} not recognized")
        random_actions.append(choice)
    return random_actions


def order_stats(X):
    _, idx, cnt = np.unique(X, return_inverse=True, return_counts=True)
    obs = np.cumsum(cnt)  # Need to do it this way due to ties
    o_stats = obs[idx]
    return o_stats


def copula_standardize(X):
    X = np.nan_to_num(np.asarray(X))  # Replace inf by something large
    assert X.ndim == 1 and np.all(np.isfinite(X))
    o_stats = order_stats(X)
    quantile = np.true_divide(o_stats, len(X) + 1)
    X_ss = ss.norm.ppf(quantile)
    return X_ss


def derive_reward(y):
    higher_is_better = any(i < 0 for i in y)
    y_std = copula_standardize(y)
    # negate score when lower is better to make reward
    rewards = torch.tensor([x if higher_is_better else -x for x in y_std])
    rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)
    return rewards


def compute_grad_norm(agent):
    grad_norm = 0.0
    for name, params in agent.named_parameters():
        if params.grad is not None:
            param_norm = params.grad.data.norm(2)
            grad_norm += param_norm.item() ** 2
    grad_norm = grad_norm ** 0.5
    return grad_norm


class MetaLearnSwarmOptimizer(AbstractOptimizer):
    primary_import = "meta-ml"

    def __init__(
        self,
        api_config,
        hidden_size=64,
        learning_rate=0.03,
        weight_decay=1.0,
        entropy_coef=0.0,
        clip_grad=1.0,
        resample_tolerance=4,
    ):
        """Build wrapper class to use optimizer in benchmark.

        Parameters
        ----------
        api_config : dict-like of dict-like
            Configuration of the optimization variables. See API description.
        """
        AbstractOptimizer.__init__(self, api_config)

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.entropy_coef = entropy_coef
        self.clip_grad = clip_grad
        self.resample_tolerance = resample_tolerance

        # initial conditions
        self.anchor_point = np.array(sample_random_actions(api_config))
        self.init_reward = np.array([0])

        # instantiate agents
        self.n_actions = len(api_config)
        self.agent = Agent(self.n_actions, hidden_size=hidden_size)
        self.optim = torch.optim.Adam(
            self.agent.parameters(),
            lr=learning_rate,
            weight_decay=self.weight_decay,
        )
        self.history = History()
        self.best_reward = float("-inf")

        # counters
        self.resample_counter = 0

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
            # TODO: next step is to implement heuristic swarm logic.
            # TODO: idea - select the location of the best-performing
            # suggestion and set that as the anchor point for the next
            # iteration.
            self.agent.train()

            prev_obs = torch.from_numpy(self.anchor_point).float()
            prev_actions, prev_reward = self.history.recall(self.n_actions)

            suggestions = []
            for i in range(n_suggestions):
                actions, log_probs, entropy, value = self.agent(
                    prev_obs, prev_actions, prev_reward
                )
                adjusted_obs = prev_obs + actions
                suggestions.append(self.make_suggestion(adjusted_obs))
                prev_actions = actions

            return suggestions
        except Exception as e:
            print(e)

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
            rewards = derive_reward(y)

            # compute loss
            self.optim.zero_grad()
            loss, memory = self.agent.update(rewards, self.entropy_coef)
            loss.backward()

            # clip gradient and compute grad norm
            nn.utils.clip_grad_norm_(self.agent.parameters(), self.clip_grad)
            grad_norm = compute_grad_norm(self.agent)
            self.optim.step()

            # resample anchor point if best reward hasn't increased
            if rewards.max() > self.best_reward:
                self.best_reward = rewards.max()
                self.resample_counter = 0
            else:
                self.resample_counter += 1

            if self.resample_counter >= self.resample_tolerance:
                print("[resetting anchor point]")
                self.best_reward = float("-inf")
                self.anchor_point = np.array(
                    sample_random_actions(self.api_config)
                )

            print(
                {
                    "mean_y": f"{np.mean(y):0.04f}",
                    "max_reward": f"{rewards.max():0.04f}",
                    "best_reward": f"{self.best_reward:0.04f}",
                    "loss": f"{loss.detach().item():0.04f}",
                    "grad_norm": f"{grad_norm:0.04f}",
                },
                f"\n{'-' * 30}",
            )
        except Exception as e:
            print(e)
        finally:
            self.history.record(memory, X, y, rewards.tolist())

    def make_suggestion(self, adjusted_obs):
        suggestion = {}
        for setting, (param, config) in zip(
            adjusted_obs, self.api_config.items()
        ):
            if config["type"] == "bool":
                min_val, max_val = 0, 1
            elif config["type"] == "cat":
                min_val, max_val = 0, len(config["values"]) - 1
            else:
                min_val, max_val = config["range"]

            # make sure setting is within bounds
            setting = (
                max(min_val, setting.detach().item())
                if setting < min_val
                else min(max_val, setting.detach().item())
            )

            if config["type"] in {"bool", "int"}:
                setting = int(setting)
            elif config["type"] == "cat":
                setting = config["values"][int(setting)]
            suggestion[param] = setting

        return suggestion


if __name__ == "__main__":
    experiment_main(MetaLearnSwarmOptimizer)
