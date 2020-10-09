"""SWARM Agent."""

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
    """A single agent."""

    def __init__(
        self,
        n_actions,
        hidden_size,
        max_cholesky_size=1000,
        dropout=0.01,
        policy_lowrank_normal=False,
    ):
        super(Agent, self).__init__()

        self.hidden_size = hidden_size
        self.n_actions = n_actions
        self.dropout = dropout
        self.max_cholesky_size = max_cholesky_size
        self.policy_lowrank_normal = policy_lowrank_normal

        # input is observation is composed of:
        # - hyperparameter anchors of the same dim as n_actions
        # - the agent's actions with range (-1, 1)
        # - the scalar reward

        # TODO: add non-linearities here
        self.encoder = nn.Sequential(
            nn.Linear(n_actions, hidden_size),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
        )

        # policy parameter layers
        self.mu = nn.Sequential(
            nn.Linear(hidden_size, n_actions),
            nn.Dropout(dropout),
            nn.LayerNorm(n_actions),
            nn.Tanh(),
        )
        self.mu_scaler = nn.Sequential(
            nn.Linear(hidden_size, n_actions),
            nn.Dropout(dropout),
            nn.LayerNorm(n_actions),
            nn.Softplus(),
        )
        if policy_lowrank_normal:
            self.cov_factor = nn.Sequential(
                nn.Linear(hidden_size, n_actions),
                nn.Dropout(dropout),
                nn.LayerNorm(n_actions),
                nn.Sigmoid(),
            )
            self.cov_diag = nn.Sequential(
                nn.Linear(hidden_size, n_actions),
                nn.Dropout(dropout),
                nn.LayerNorm(n_actions),
                nn.Sigmoid(),
            )
            self.distribution = D.LowRankMultivariateNormal
        else:
            self.cov = nn.Sequential(
                nn.Linear(hidden_size, n_actions),
                nn.Dropout(dropout),
                nn.LayerNorm(n_actions),
                nn.Sigmoid(),
            )
            self.distribution = D.MultivariateNormal

        self.cov_scaler = nn.Sequential(
            nn.Linear(hidden_size, n_actions),
            nn.Dropout(dropout),
            nn.LayerNorm(n_actions),
            nn.Sigmoid(),
        )

        self.gp_model = None
        self.hypers = None

    def update_gp(self, obs, actions, rewards, num_steps=100):
        with gpytorch.settings.max_cholesky_size(self.max_cholesky_size):
            self.gp_model = train_gp_model(
                X=torch.cat([obs, actions, obs + actions], dim=1).double(),
                # X=torch.cat([obs + actions], dim=1).double(),
                y=rewards.double(),
                num_steps=num_steps,
                hypers=self.hypers,
            )
            # save gp hyperparameters
            self.hypers = self.gp_model.state_dict()

    def get_value(self, obs, actions, adjusted_obs):
        self.gp_model.eval()
        self.gp_model.likelihood.eval()
        with torch.no_grad(), gpytorch.settings.max_cholesky_size(
            self.max_cholesky_size
        ):
            X = torch.cat([
                torch.repeat_interleave(
                    obs.unsqueeze(0), actions.shape[0], dim=0
                ),
                actions,
                adjusted_obs,
            ], dim=1).double()
            # X = adjusted_obs.double()
            value_dist = self.gp_model.likelihood(self.gp_model(X))
            # TODO: do something with the stddev of the distribution
            return value_dist.mean
            # if batch_size == 1:
            #     return value_dist.mean
            # else:
            #     return value_dist.sample(torch.Size([batch_size])).t()

    def forward(self, obs, n_candidates, action_scaler=1.0, cov_scaler=1.0):
        """
        Parameters:
            obs: tensor of observation
            n: generate this number of actions
            cov_scaler: manually scales the actions

        Returns:
            actions, adjusted observations, action log probabilities, entropy,
            and value.
        """
        encoded = self.encoder(obs)
        actions, log_probs, entropy = self.act(
            encoded, n_candidates, action_scaler, cov_scaler,
        )
        adjusted_obs = torch.clamp(obs + actions, 0, 1)
        value = self.get_value(obs, actions, adjusted_obs)
        return actions, adjusted_obs, log_probs, entropy.unsqueeze(0), value

    def prob_dist(self, encoded, mu_scaler=1.0, cov_scaler=1.0):
        if self.policy_lowrank_normal:
            return self.distribution(
                self.mu(encoded) * self.mu_scaler(encoded),
                (
                    self.cov_factor(encoded)
                    * self.cov_scaler(encoded)
                ).view(-1, 1),
                self.cov_diag(encoded),
            )
        else:
            return self.distribution(
                self.mu(encoded) * self.mu_scaler(encoded),
                (
                    self.cov(encoded).diag()
                    * self.cov_scaler(encoded)
                    * cov_scaler
                ),
            )

    def act(self, encoded, n_candidates, action_scaler=1.0, cov_scaler=1.0):
        prob_dist = self.prob_dist(encoded, cov_scaler=cov_scaler)
        actions = prob_dist.rsample(torch.Size([n_candidates])) * action_scaler
        log_probs = prob_dist.log_prob(actions)
        return actions, log_probs, prob_dist.entropy()

    def evaluate_experiences(self, obs, actions):
        """Policy and critic evaluates past actions."""
        encoded = self.encoder(obs)
        prob_dist = self.prob_dist(encoded)
        log_probs = prob_dist.log_prob(actions)
        adjusted_obs = torch.clamp(obs + actions, 0, 1)
        value = self.get_value(obs, actions, adjusted_obs)
        return value.squeeze(), log_probs.squeeze(), prob_dist.entropy()

    def update(self, rewards, entropy_coef=0.0):
        advantage = rewards - torch.stack(self.memory.values)
        actor_loss = torch.stack(self.memory.log_probs) * advantage
        critic_loss = 0.5 * advantage ** 2
        entropy_loss = torch.stack(self.memory.entropies) * entropy_coef
        loss = actor_loss.mean() + critic_loss.mean() - entropy_loss.mean()
        return loss, self.memory.detach()


if __name__ == "__main__":
    # TODO: turn this into a set of unit test
    from typing import Dict

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
