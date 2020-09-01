"""Metalearn Agent."""

from copy import copy
from typing import List, Any, Union

import gpytorch
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D

from custom_typing import Array
from gaussian_process import train_gp_model


def fill_tril_2d_square(fill, values):
    n = fill.shape[0]
    indices = torch.tril_indices(n, n, -1)
    assert len(values) == indices.shape[1]
    for idx in range(indices.shape[1]):
        i, j = indices[:, idx]
        fill[i, j] = values[idx]
    return fill


class Agent(nn.Module):
    """A single metalearning agent."""

    def __init__(
        self, n_actions, hidden_size, max_cholesky_size=1000, dropout=0.01,
    ):
        super(Agent, self).__init__()

        self.hidden_size = hidden_size
        self.n_actions = n_actions
        self.dropout = dropout
        self.max_cholesky_size = max_cholesky_size

        # input is observation is composed of:
        # - hyperparameter anchors of the same dim as n_actions
        # - the agent's actions with range (-1, 1)
        # - the scalar reward
        self.encoder = nn.Sequential(
            nn.Linear(n_actions, hidden_size),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_size),
        )

        self.critic = nn.Sequential(
            nn.Linear(hidden_size + (n_actions * 3), hidden_size),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, 1),
        )

        # policy parameter layers
        self.cov_factor = nn.Sequential(
            nn.Linear(hidden_size, n_actions),
            nn.Dropout(dropout),
            nn.LayerNorm(n_actions),
            nn.Hardsigmoid(),
        )
        self.cov_diag = nn.Sequential(
            nn.Linear(hidden_size, n_actions),
            nn.Dropout(dropout),
            nn.LayerNorm(n_actions),
            nn.Hardsigmoid(),
        )

        self.distribution = D.LowRankMultivariateNormal
        self.memory = Memory()
        self.gp_model = None
        self.hypers = None

    def update_gp(self, obs, actions, rewards, num_steps=100):
        with gpytorch.settings.max_cholesky_size(self.max_cholesky_size):
            self.gp_model = train_gp_model(
                X=torch.cat([obs, actions, obs + actions], dim=1).double(),
                y=rewards.double(),
                num_steps=num_steps,
                hypers=self.hypers,
            )
            # save gp hyperparameters
            self.hypers = self.gp_model.state_dict()

    def get_value(self, obs, actions, adjusted_obs):
        adjusted_obs = torch.clamp(obs + actions, 0, 1)
        self.gp_model.eval()
        self.gp_model.likelihood.eval()
        with torch.no_grad(), gpytorch.settings.max_cholesky_size(
            self.max_cholesky_size
        ):
            value_dist = self.gp_model.likelihood(
                self.gp_model(
                    torch.cat([obs, actions, adjusted_obs], dim=0)
                    .double()
                    .view(1, -1)
                )
            )
            return value_dist.sample(torch.Size([1])).squeeze()

    def forward(self, obs, cov_scaler=1.0):
        encoded = self.encoder(obs)
        actions, log_probs, entropy, prob_dist = self.act(encoded, cov_scaler)
        adjusted_obs = torch.clamp(obs + actions, 0, 1)
        value = self.get_value(obs, actions, adjusted_obs)
        self.memory.record(
            obs, actions, adjusted_obs, log_probs, entropy, value, prob_dist
        )
        return actions, adjusted_obs, log_probs, entropy, value, prob_dist

    def prob_dist(self, encoded):
        return self.distribution(
            torch.zeros(self.n_actions),
            self.cov_factor(encoded).view(-1, 1),
            self.cov_diag(encoded),
        )

    def act(self, encoded, cov_scaler=1.0):
        prob_dist = self.prob_dist(encoded)
        actions = prob_dist.rsample() * cov_scaler
        log_probs = prob_dist.log_prob(actions)
        return actions, log_probs, prob_dist.entropy(), prob_dist

    def evaluate_actions(self, obs, actions):
        adjusted_obs = torch.clamp(obs + actions, 0, 1)
        value = self.get_value(obs, actions, adjusted_obs)
        return value

    def evaluate_experiences(self, obs, actions):
        """Policy and critic evaluates past actions."""
        encoded = self.encoder(obs)
        prob_dist = self.prob_dist(encoded)
        log_probs = prob_dist.log_prob(actions)
        adjusted_obs = torch.clamp(obs + actions, 0, 1)
        value = self.get_value(obs, actions, adjusted_obs)
        return value, log_probs, prob_dist.entropy()

    def update(self, rewards, entropy_coef=0.0):
        advantage = rewards - torch.stack(self.memory.values)
        actor_loss = torch.stack(self.memory.log_probs) * advantage
        critic_loss = 0.5 * advantage ** 2
        entropy_loss = torch.stack(self.memory.entropies) * entropy_coef
        loss = actor_loss.mean() + critic_loss.mean() - entropy_loss.mean()
        return loss, self.memory.detach()


class Memory:
    def __init__(
        self,
        obs=None,
        actions=None,
        adjusted_obs=None,
        log_probs=None,
        entropies=None,
        values=None,
        prob_dists=None,
    ):
        self.obs: List[List[Array]] = [] if obs is None else obs
        self.actions: List[List[Array]] = [] if actions is None else actions
        self.adjusted_obs: List[
            List[Array]
        ] = [] if adjusted_obs is None else adjusted_obs
        self.log_probs: List[Array] = [] if log_probs is None else log_probs
        self.entropies: List[Array] = [] if entropies is None else entropies
        self.values: List[List[Array]] = [] if values is None else values
        self.prob_dists: List[Array] = [] if prob_dists is None else prob_dists

    def items(self):
        return {
            "obs": self.obs,
            "actions": self.actions,
            "adjusted_obs": self.adjusted_obs,
            "log_probs": self.log_probs,
            "entropies": self.entropies,
            "values": self.values,
            "prob_dists": [copy(x) for x in self.prob_dists],
        }.items()

    def record(
        self, obs, action, adjusted_obs, log_probs, entropy, value, prob_dists
    ):
        self.obs.append(obs)
        self.actions.append(action)
        self.adjusted_obs.append(adjusted_obs)
        self.log_probs.append(log_probs)
        self.entropies.append(entropy)
        self.values.append(value)
        self.prob_dists.append(prob_dists)

    def erase(self):
        del self.obs[:]
        del self.actions[:]
        del self.adjusted_obs[:]
        del self.log_probs[:]
        del self.entropies[:]
        del self.values[:]
        del self.prob_dists[:]

    def detach(self):
        memory = Memory(
            *[
                [
                    x if isinstance(x, np.ndarray) else x.detach().numpy()
                    for x in mem_list
                ]
                for mem_list in (
                    self.obs,
                    self.actions,
                    self.adjusted_obs,
                    self.log_probs,
                    self.entropies,
                    self.values,
                )
                if mem_list is not None
            ]
            + [self.prob_dists]
        )
        self.erase()
        return memory


if __name__ == "__main__":
    # TODO: turn this into a set of unit test
    from typing import Dict
    import numpy as np

    np.random.seed(1001)

    api_config: Dict[str, Any] = {
        "max_depth": {"type": "int", "space": "linear", "range": (1, 15)},
        "min_samples_split": {
            "type": "real",
            "space": "logit",
            "range": (0.01, 0.99),
        },
        "min_impurity_decrease": {
            "type": "real",
            "space": "linear",
            "range": (0.0, 0.5),
        },
    }

    entropy_coef = 0.1
    clip_grad = 10.0

    random_actions: List = []
    for param, config in api_config.items():
        min_val, max_val = config["range"]
        choice = (
            np.random.randint(min_val, max_val + 1)
            if config["type"] == "int"
            else np.random.uniform(min_val, max_val)
            if config["type"] == "real"
            else np.random.choice([0, 1])
            if config["type"] == "bool"
            else np.random.choice(config["values"])
            if config["type"] == "cat"
            else None
        )
        if choice is None:
            raise ValueError(f"type {config['type']} not recognized")
        random_actions.append(choice)

    init_obs = torch.tensor(random_actions).float()
    init_actions = torch.tensor(
        [0 for _ in range(len(random_actions))]
    ).float()
    init_reward = torch.tensor([0]).float()

    # create agent and optimizer
    agent = Agent(n_actions=len(api_config), hidden_size=5)
    optim = torch.optim.Adam(agent.parameters(), lr=0.1, weight_decay=0.1)

    # one forward pass
    agent.train()
    actions, log_probs, entropy, value = agent.forward(
        init_obs, init_actions, init_reward
    )

    # prepare suggestion
    adjusted_obs = init_obs + actions
    # make sure it's in range
    suggestion = {}
    for setting, (param, config) in zip(adjusted_obs, api_config.items()):
        min_val, max_val = config["range"]
        setting = (
            max(min_val, setting.detach().item())
            if setting < min_val
            else min(max_val, setting.detach().item())
        )
        if config["type"] == "int":
            setting = int(setting)
        suggestion[param] = setting

    # compute loss and update
    optim.zero_grad()
    mock_reward = torch.tensor([1])
    advantage = mock_reward - value

    actor_loss = log_probs * advantage
    critic_loss = 0.5 * advantage ** 2

    loss = actor_loss + critic_loss - (entropy * entropy_coef).mean()
    loss.backward()
    nn.utils.clip_grad_norm_(agent.parameters(), clip_grad)

    grad_norm = 0.0
    for name, params in agent.named_parameters():
        if params.grad is not None:
            param_norm = params.grad.data.norm(2)
            grad_norm += param_norm.item() ** 2
    grad_norm = grad_norm ** 0.5

    print(f"grad norm: {grad_norm}")
    optim.step()
