"""Training Routine for multiple agents."""

import itertools
from collections import defaultdict, namedtuple
from copy import deepcopy
from typing import Any, List, Dict

import numpy as np
import torch
from torch.quasirandom import SobolEngine

from agent import Agent
from custom_typing import Array
from utils import derive_reward


# TODO: implement MetaAgent that manages the LocalAgents so as to
# maximize reward.
# - its inputs are the value estimates from the local agents
#   - distance from anchor point
#   - previous reward
#   - hyperparameter space metafeatures
#   - num range
#   - max range
#   - n numerical
#   - n categorical
#   - n boolean
# - it should output a value estimate of its own, which is used to
#   rank the candidates for selection
#   - maybe it should estimate the error between reward and the
#     collective value estimate?

# TODO:
# - try constraining cov matrix and adding learnable mu parameter for policy
# - modify Agent to produce batch_size of actions in the forward pass

Candidates = namedtuple(
    "Candidates", [
        "agent_id",
        "obs",
        "actions",
        "adjusted_obs",
        "log_probs",
        "entropy",
        "value",
    ]
)

Experiences = namedtuple(
    "Experiences", ["outcomes", "agent_ids", "obs", "actions"]
)

ReplayResults = namedtuple(
    "ReplayResults", [
        "outcomes",
        "agent_ids",
        "obs",
        "actions",
        "values",
        "log_probs",
        "entropies",
    ]
)


class Swarm:
    def __init__(
        self,
        n_agents,
        agent_config,
        anchor_fn,
        optim,
        optim_config,
        num_gp_update_steps=100,
        n_perturb=3,
        noise_window=0.1,
        random_seed=43,
    ):
        self.n_agents = n_agents
        self.n_actions = agent_config["n_actions"]
        self.anchor_fn = anchor_fn
        self.num_gp_update_steps = num_gp_update_steps
        self.n_perturb = n_perturb
        self.noise_window = noise_window
        self.random_seed = random_seed

        # initialize agents
        self.local_agents = [
            LocalAgent(Agent(**agent_config), i) for i in range(n_agents)
        ]
        self.optim = optim(
            [
                {"params": local_agent.agent.parameters()}
                for local_agent in self.local_agents
            ],
            **optim_config,
        )
        self.collective_memory = None
        self.history = History()
        self.n_successes = 0
        self.n_failures = 0
        self.n_updates = 0

        # keep track of best points so far
        self.best_obs = None
        self.best_reward = None
        self.best_y = None

    def init_suggestions(self, batch_size):
        """Initial suggestion using random points."""

        # init_obs = [
        #     torch.from_numpy(self.anchor_fn()).float()
        #     for _ in range(batch_size)
        # ]
        sobol = SobolEngine(
            self.n_actions, scramble=True, seed=self.random_seed
        )
        obs_sample = sobol.draw(batch_size)
        init_obs = [
            obs_sample[i, :] for i in range(batch_size)
        ]
        init_actions = [
            torch.randn(self.n_actions) for _ in range(batch_size)
        ]

        # initial actions attributed to 0th agent to start
        self.init_obs = init_obs
        self.init_actions = init_actions
        torch.stack(init_actions)
        return torch.clamp(
            torch.stack(init_obs) + torch.stack(init_actions),
            0, 1
        )

    def init_update(self, y, rewards, entropy_coef):
        for i, local_agent in enumerate(self.local_agents):
            # update GP
            experiences = defaultdict(list)
            for obs, action, reward in zip(
                self.init_obs,
                self.init_actions,
                rewards,
            ):
                experiences["obs"].append(obs)
                experiences["actions"].append(action)
                experiences["rewards"].append(reward)

            local_agent.agent.update_gp(
                torch.stack(experiences["obs"]),
                torch.stack(experiences["actions"]),
                torch.stack(experiences["rewards"]),
                self.num_gp_update_steps,
            )

        best_idx = torch.argmax(rewards).item()
        self.best_reward = rewards[best_idx].item()
        self.best_y = y[best_idx]
        self.best_obs = self.init_obs[torch.argmax(rewards).item()]

        init_obs = torch.stack(self.init_obs)
        init_actions = torch.stack(self.init_actions)
        init_suggestions = init_obs + init_actions
        self.history.record(
            CollectiveMemory(
                # initial actions attributed to 0th agent to start
                agent_ids=torch.from_numpy(
                    np.random.choice([x.id for x in self.local_agents], len(y))
                ),
                obs=init_obs,
                actions=init_actions,
                adjusted_obs=init_suggestions,
            ).detach(),
            y,
        )
        return None

    @property
    def action_scaler(self):
        return 1
        # return 1 / (1 + self.n_successes * 0.25)

    @property
    def cov_scaler(self):
        return 1
        # return 1 / (1 + self.n_successes * 0.5)

    def __call__(self, batch_size, n_candidates=10, n_iter=1):
        if self.n_updates == 0:
            return self.init_suggestions(batch_size)

        print(f"[cov scaler] {self.cov_scaler}")
        # sample the local agents for each candidate
        cand_batches: List[Candidates] = []
        for _ in range(n_iter):
            for local_agent in self.local_agents:
                cand_batches.append(
                    local_agent.create_candidates(
                        (
                            self.best_obs
                            if local_agent.best_obs is None
                            else local_agent.best_obs
                        ),
                        n_candidates,
                        action_scaler=self.action_scaler,
                        cov_scaler=self.cov_scaler,
                    )
                )

        # collect all candidates from all agents
        candidates: Dict[str, Array] = {}
        for field, tensor_list in zip(Candidates._fields, zip(*cand_batches)):
            candidates[field] = torch.cat(tensor_list)

        # get suggestions
        values = candidates["value"].clone()
        best_indices = torch.argsort(values, descending=True)[:batch_size]
        # best_indices = []
        # for i in range(batch_size):
        #     best_index = torch.argmax(values[:, i])
        #     best_indices.append(best_index.item())
        #     # make sure same candidate isn't picked twice
        #     values[best_index, :] = float("-inf")

        suggestions = candidates["adjusted_obs"][best_indices]

        self.collective_memory = CollectiveMemory(
            agent_ids=candidates["agent_id"][best_indices],
            obs=candidates["obs"][candidates["agent_id"][best_indices]],
            actions=candidates["actions"][best_indices],
            adjusted_obs=suggestions,
            log_probs=candidates["log_probs"][best_indices],
            entropies=candidates["entropy"][
                candidates["agent_id"][best_indices]
            ],
            # values=candidates["value"][best_indices, range(batch_size)],
            values=candidates["value"][best_indices],
        )
        # import ipdb; ipdb.set_trace()
        return suggestions

    def update(self, y, entropy_coef=0.0):
        self.n_updates += 1

        if self.n_updates <= 1:
            rewards = derive_reward(y)
            return self.init_update(y, rewards, entropy_coef)

        replay: ReplayResults = self.experience_replay()

        # construct global dataset
        global_y = replay.outcomes + y
        values = torch.cat([replay.values, self.collective_memory.values])
        agent_ids = torch.cat(
            [replay.agent_ids, self.collective_memory.agent_ids]
        )
        obs = torch.cat([replay.obs, self.collective_memory.obs])
        actions = torch.cat(
            [replay.actions, self.collective_memory.actions]
        )
        log_probs = torch.cat(
            [replay.log_probs, self.collective_memory.log_probs]
        )
        entropies = torch.cat(
            [replay.entropies, self.collective_memory.entropies]
        )
        rewards = derive_reward(global_y)

        # compute actor losses based on current and past actions
        advantage = rewards - values
        actor_loss = log_probs * advantage
        entropy_loss = entropies * entropy_coef
        loss = actor_loss.mean() - entropy_loss.mean()

        # update value approximator GPs
        for local_agent in self.local_agents:
            idx = agent_ids == local_agent.id
            local_agent.agent.update_gp(
                obs[idx], actions[idx].detach(), rewards[idx],
                self.num_gp_update_steps,
            )

        # update reward, success and failure states
        self.update_states(global_y, rewards, obs, agent_ids)
        self.history.record(
            self.collective_memory.detach(), y,
        )
        return loss

    def experience_replay(self):
        if len(self.history) == 0:
            return None

        exp: Experiences = self.history.recall(
            n_perturb=self.n_perturb,
            noise_window=self.noise_window,
            random_seed=self.random_seed
        )
        replay_results = defaultdict(list)

        for local_agent in self.local_agents:
            idx = torch.where(exp.agent_ids == local_agent.id)[0]
            if len(idx) == 0:
                continue
            values, log_probs, entropies = [
                torch.stack(x) for x in
                zip(*[
                    local_agent.agent.evaluate_experiences(
                        exp.obs[i], exp.actions[[i], :]
                    )
                    for i in idx
                ])
            ]
            replay_results["outcomes"].extend(
                [o for i, o in enumerate(exp.outcomes) if i in idx]
            )
            replay_results["agent_ids"].append(
                exp.agent_ids[idx].long()
            )
            replay_results["obs"].append(exp.obs[idx])
            replay_results["actions"].append(exp.actions[idx])
            replay_results["values"].append(values)
            replay_results["log_probs"].append(log_probs)
            replay_results["entropies"].append(entropies)

        return ReplayResults(**{
            k: torch.cat(v) if isinstance(v[0], torch.Tensor) else v
            for k, v in replay_results.items()
        })

    def update_states(self, y, rewards, obs, agent_ids):
        y = torch.tensor(y)
        best_idx = torch.argmin(y)
        best_y, best_reward = y[best_idx], rewards[best_idx]

        # assume lower scores are better
        if best_y < self.best_y:
            print(f"[updating best observation] {self.best_y} >> {best_y}")
            self.best_reward = best_reward
            self.best_y = best_y
            self.best_obs = obs[best_idx].detach()
            self.n_successes += 1
        else:
            self.n_failures += 1

        # update local agent best states
        for local_agent in self.local_agents:
            idx = agent_ids == local_agent.id
            local_y, local_rewards, local_obs = y[idx], rewards[idx], obs[idx]
            best_local_idx = torch.argmin(local_y)
            if local_y[best_local_idx] < local_agent.best_y:
                local_agent.best_y = local_y[best_local_idx]
                local_agent.best_reward = local_rewards[best_local_idx]
                local_agent.best_obs = local_obs[best_local_idx]


class CollectiveMemory:
    def __init__(
        self,
        agent_ids: Array,
        obs: Array,
        actions: Array,
        adjusted_obs: Array,
        log_probs: Array = None,
        entropies: Array = None,
        values: Array = None,
    ):
        self.agent_ids: Array = agent_ids
        self.obs: Array = obs
        self.actions: Array = actions
        self.adjusted_obs: Array = adjusted_obs
        self.log_probs: Array = None if log_probs is None else log_probs
        self.entropies: Array = None if entropies is None else entropies
        self.values: Array = None if values is None else values

    def detach(self):

        def _detach(x):
            return x if x is None else deepcopy(x.detach().numpy())

        memory = CollectiveMemory(
            _detach(self.agent_ids),
            _detach(self.obs),
            _detach(self.actions),
            _detach(self.adjusted_obs),
            _detach(self.log_probs),
            _detach(self.entropies),
            _detach(self.values),
        )
        return memory

    def __len__(self):
        return len(self.agent_ids)


class History:
    def __init__(self):
        self.collective_memories: List[CollectiveMemory] = []
        self.outcomes: List[List[float]] = []

    def record(self, cmemory, outcomes):
        self.collective_memories.append(deepcopy(cmemory))
        self.outcomes.append(deepcopy(outcomes))

    def recall(self, n_perturb=0, noise_window=0.1, random_seed=None):
        """Iterate through memories with most recent ones first.

        If n_perturb > 0, this method returns perturbed experiences that
        introduce random noise to the historical experiences.
        """
        agent_ids, obs, actions, outcomes = [], [], [], []

        # TODO: perturb observations and compute the difference to obtain
        # pert_action = (obs + action) - pert_obs.
        # This should result in "noisy recall" that maps to the same outcome.

        for cmemory, outcome in zip(self.collective_memories, self.outcomes):
            if n_perturb == 0:
                agent_ids.append(cmemory.agent_ids)
                obs.append(cmemory.obs)
                actions.append(cmemory.actions)
                outcomes.extend(outcome)
            else:
                sobol = SobolEngine(
                    cmemory.obs.shape[1], scramble=True, seed=random_seed
                )
                for i in range(cmemory.obs.shape[0]):
                    noise = sobol.draw(n_perturb).numpy()
                    window = noise_window / 2
                    lb = np.clip(cmemory.obs[i] - window, 0, 1)
                    ub = np.clip(cmemory.obs[i] + window, 0, 1)
                    pert_obs = lb + (ub - lb) * noise
                    pert_actions = (
                        cmemory.obs[i] + cmemory.actions[i]
                        - pert_obs
                    )
                    agent_ids.append(
                        np.array(
                            [cmemory.agent_ids[i] for _ in range(n_perturb)]
                        )
                    )
                    obs.append(pert_obs)
                    actions.append(pert_actions)
                    outcomes.extend([outcome[i] for _ in range(n_perturb)])
        return Experiences(
            outcomes,
            torch.from_numpy(np.concatenate(agent_ids)),
            torch.from_numpy(np.concatenate(obs)).float(),
            torch.from_numpy(np.concatenate(actions)).float(),
        )

    def erase(self):
        del self.collective_memories[:]
        del self.outcomes[:]

    def __getitem__(self, i):
        return self.collective_memories[i], self.outcomes[i]

    def __len__(self):
        return len(self.collective_memories)


class LocalAgent:
    def __init__(self, agent, id):
        self.agent = agent
        self.id = id
        self.best_y = float("inf")
        self.best_reward = float("-inf")
        self.best_obs = None

    def create_candidates(
        self, obs, n_candidates, action_scaler=1.0, cov_scaler=1.0
    ):
        self.agent.train()
        return Candidates(
            torch.tensor([self.id for _ in range(n_candidates)]),
            obs.view(1, -1),
            *self.agent(obs, n_candidates, action_scaler, cov_scaler)
        )
