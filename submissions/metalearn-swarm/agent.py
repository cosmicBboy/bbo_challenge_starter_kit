"""Metalearn Agent."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D


class Agent(nn.Module):
    """A single metalearning agent."""

    def __init__(self, input_size, hidden_size, n_actions):
        super(MetaLearnAgent, self).__init__()

        self.hidden_size = hidden_size
        self.n_actions = n_actions
        # input is observation is composed of:
        # - the hyperparameter setting anchors of the same dimensionality as
        #   n_actions, then the agents
        # - the agent's actions, which is a value centered around 0 with a
        #   range of -1 and 1, which are added to the anchors for env
        #   evaluation
        # - the scalar reward
        self.input_size = (n_actions * 2) + 1

        self.encoder = nn.Linear(self.input_size, hidden_size)
        self.critic = nn.Sequential(
            nn.Linear(hidden_size, hidden_size), nn.Linear(hidden_size, 1),
        )

        self.mu = nn.Sequential(nn.Linear(hidden_size, n_actions), nn.Tanh())
        self.sigma = nn.Sequential(
            nn.Linear(hidden_size, n_actions), nn.Softplus()
        )
        self.distribution = D.MultivariateNormal

    def forward(self, obs, prev_actions, prev_reward):
        encoded = self.encoder(torch.cat([obs, prev_actions, prev_reward]))
        actions, log_probs, entropy = self.act(encoded)
        value = self.critic(encoded)
        return actions, log_probs, entropy, value

    def act(self, encoded):
        prob_dist = self.distribution(
            self.mu(encoded), self.sigma(encoded).diag(),
        )
        actions = prob_dist.rsample()
        return actions, prob_dist.log_prob(actions), prob_dist.entropy()

    def update(self):
        pass


if __name__ == "__main__":
    from typing import Any, List, Dict
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
    agent = Agent(input_size=3, hidden_size=5, n_actions=len(api_config))
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

    # TODO:
    # - move optimization logic to optimizer.py
    # - implement basic logic for generating suggestions:
    #   - for each suggestion batch, generate n random initial points
    #   - for each initial point, the agent generates m action adjustments as
    #     a function of the hyperparameter values, the previous adjustments,
    #     and the previous reward.
    #   - this procedure results in n x m candidates
    #   - rank them based on predicted value, and select top n_suggestions
    #   - update controller based on rewards of the selected candidates
    #
    # - implement swarm logic for generating suggestions:
    #   - create a local agent for each of n randomly initialized points
    #   - global agent is a metalearning agent that determines which
    #     candidate to select from the m actions for each of n agents.
    #   - given the suggestion, anchor observation and adjustment action,
    #     this global agent produces an estimate of the value and a probability
    #     between 0 and 1 to decide whether or not to include the candidate.
    #   - vanilla version can be a FFN that produces estimates sequentially,
    #     until n_suggestions have been generated.
    #   - the metalearning version would be an RNN that processes the rewards
    #     that decodes the rewards and actions, stopping when n_suggestions
    #     have been generated.
    #   - critic loss function would be reward - value estimate error
    #   - actor loss function would be log probability of "select candidate"
    #     action.
    #   - crazy idea: maybe use the global agent to further train the local
    #     agents by feeding them the reward estimates as the reward for
    #     those candidates that were not selected. This could potentially
    #     introduce a lot of bias into the system if the global agent's value
    #     estimates are really off.
    #
    # - idea: to "refresh" the random initial points, every t turns randomly
    #   perturb them.
