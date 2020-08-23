"""Metalearn Agent."""

from typing import List, NamedTuple, Any, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D


Array = Union[torch.Tensor, np.ndarray]


class Agent(nn.Module):
    """A single metalearning agent."""

    def __init__(self, n_actions, hidden_size):
        super(Agent, self).__init__()

        self.hidden_size = hidden_size
        self.n_actions = n_actions

        # input is observation is composed of:
        # - hyperparameter anchors of the same dim as n_actions
        # - the agent's actions with range (-1, 1)
        # - the scalar reward
        self.input_size = (n_actions * 2) + 1

        self.encoder = nn.Sequential(
            nn.Linear(self.input_size, hidden_size), nn.LayerNorm(hidden_size),
        )

        self.critic = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, 1),
        )

        # policy parameter layers
        self.mu = nn.Sequential(
            nn.Linear(hidden_size, n_actions),
            nn.LayerNorm(n_actions),
            nn.Tanh(),
        )
        self.cov_factor = nn.Sequential(
            nn.Linear(hidden_size, n_actions),
            nn.LayerNorm(n_actions),
            nn.Hardsigmoid(),
        )
        self.cov_diag = nn.Sequential(
            nn.Linear(hidden_size, n_actions),
            nn.LayerNorm(n_actions),
            nn.Hardsigmoid(),
        )
        self.distribution = D.LowRankMultivariateNormal
        # self.distribution = D.MultivariateNormal
        self.memory = Memory()

    def forward(self, obs, prev_actions, prev_reward):
        encoded = self.encoder(torch.cat([obs, prev_actions, prev_reward]))
        actions, log_probs, entropy = self.act(encoded)
        value = self.critic(encoded)[0]
        self.memory.record(actions, obs + actions, log_probs, entropy, value)
        return actions, log_probs, entropy, value

    def act(self, encoded):
        prob_dist = self.distribution(
            self.mu(encoded) * 0.5,
            self.cov_diag(encoded).view(-1, 1),
            self.cov_factor(encoded),
        )
        actions = prob_dist.rsample()

        log_probs = prob_dist.log_prob(actions)
        return actions, log_probs, prob_dist.entropy()

    def update(self, rewards, entropy_coef=0.0):
        advantage = rewards - torch.stack(self.memory.values)
        actor_loss = torch.stack(self.memory.log_probs) * advantage
        critic_loss = 0.5 * advantage ** 2
        entropy_loss = torch.stack(self.memory.entropies) * entropy_coef
        loss = actor_loss.mean() + critic_loss.mean() - entropy_loss.mean()
        return loss, self.memory.detach()


class Memory(NamedTuple):
    actions: List[List[Array]] = []
    adjusted_obs: List[List[Array]] = []
    log_probs: List[Array] = []
    entropies: List[Array] = []
    values: List[Array] = []

    def record(self, action, adjusted_obs, log_probs, entropy, value):
        self.actions.append(action)
        self.adjusted_obs.append(adjusted_obs)
        self.log_probs.append(log_probs)
        self.entropies.append(entropy)
        self.values.append(value)

    def erase(self):
        del self.actions[:]
        del self.adjusted_obs[:]
        del self.log_probs[:]
        del self.entropies[:]
        del self.values[:]

    def detach(self):
        memory = Memory(
            *(
                [x.detach().numpy() for x in mem_list]
                for mem_list in (
                    self.actions,
                    self.adjusted_obs,
                    self.log_probs,
                    self.entropies,
                    self.values,
                )
            )
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
