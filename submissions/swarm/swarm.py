"""Training Routine for multiple agents."""

from collections import defaultdict
from copy import deepcopy
from typing import Any, List, Dict

import numpy as np
import torch

from agent import Agent, Memory
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


class Swarm:
    def __init__(
        self,
        n_agents,
        agent_config,
        anchor_fn,
        optim,
        optim_config,
        kappa=1.0,
        success_threshold=3,
        failure_tolerance=4,
    ):
        self.n_agents = n_agents
        self.n_actions = agent_config["n_actions"]
        self.anchor_fn = anchor_fn
        self.kappa = kappa
        self.success_threshold = success_threshold
        self.failure_tolerance = failure_tolerance

        # initialize agents
        self.local_agents = [
            LocalAgent(Agent(**agent_config)) for _ in range(n_agents)
        ]
        self.optim = optim(
            [
                {"params": local_agent.agent.parameters()}
                for local_agent in self.local_agents
            ],
            **optim_config,
        )
        self.collective_memory = CollectiveMemory()
        self.history = History()
        self.n_successes = 0
        self.n_failures = 0
        self.n_updates = 0

        # keep track of best points so far
        self.best_obs = None
        self.best_reward = None
        self.best_y = None

        # TODO:
        # - normalize rewards such that if rewards > 0, log-transform before
        #   negating
        # - experience replay: keep a buffer of n worst and n best obs, actions
        #   and rewards. Every update, include an off-policy evaluation of
        #   these data points per agent to re-inforce actions that led to
        #   best results and discourage actions that led to worst results

    def init_suggestions(self, batch_size):
        """Initial suggestion using random points."""
        init_obs = [
            torch.from_numpy(self.anchor_fn()).float()
            for _ in range(batch_size)
        ]
        init_actions = [torch.zeros(self.n_actions) for _ in range(batch_size)]

        evaluator_estimates = defaultdict(list)
        for obs, action in zip(init_obs, init_actions):
            local_evaluator_estimates = defaultdict(list)
            for local_agent in self.local_agents:
                local_evaluator_estimates["values"].append(
                    local_agent.agent.evaluate_actions(obs, action)
                )
            for k, v in local_evaluator_estimates.items():
                evaluator_estimates[k].append(v)

        self.init_obs = init_obs
        self.init_collective_estimates = torch.stack(
            [torch.stack(x) for x in evaluator_estimates["values"]]
        )
        return init_obs

    def init_update(self, y, rewards, entropy_coef):
        advantage = rewards.view(-1, 1) - self.init_collective_estimates
        loss = (0.5 * advantage ** 2).mean()

        best_idx = torch.argmax(rewards).item()
        self.best_reward = rewards[best_idx].item()
        self.best_y = y[best_idx]
        self.best_obs = self.init_obs[torch.argmax(rewards).item()]
        self.history.record(
            CollectiveMemory(
                # initial actions attributed to 0th agent to start
                [0 for _ in range(len(self.init_obs))],
                Memory(
                    obs=self.init_obs,
                    actions=[
                        np.zeros(self.n_actions)
                        for _ in range(len(self.init_obs))
                    ],
                    adjusted_obs=self.init_obs,
                )
            ).detach(),
            y
        )
        return loss

    def __call__(self, batch_size, n_per_agent=10):
        # TODO: enable off-policy learning by having evaluator agents generate
        # log probabilities for candidate actions.
        # - agent.evaluate_actions emits value, log_probs, and entropy
        # - in the update method, each agent should have its own actor loss

        if self.n_updates == 0:
            return self.init_suggestions(batch_size)

        # cov_scaler = 1.0
        cov_scaler = 1 / (self.n_successes + 1)
        print(f"[cov scaler] {cov_scaler}")

        suggestions = []
        for _ in range(batch_size):

            # sample the local agents for each candidate
            candidates, actions, agent_index = [], [], []
            proposer_estimates = defaultdict(list)
            for i, local_agent in enumerate(self.local_agents):
                agent_index.extend([i for _ in range(n_per_agent)])
                agent_candidates, agent_actions = (
                    local_agent.create_candidates(
                        self.best_obs, n_per_agent, cov_scaler
                    )
                )
                candidates.extend(agent_candidates)
                actions.extend(agent_actions)

                # extract memories from agent
                for field, memory in local_agent.agent.memory.items():
                    proposer_estimates[field].extend(memory)

                local_agent.agent.memory.erase()

            evaluator_estimates = defaultdict(list)
            for agent_i, cand, action in zip(agent_index, candidates, actions):
                # other agents evaluate candidates
                local_evaluator_estimates = defaultdict(list)
                evaluator = self.local_agents[agent_i]
                for j, evaluator in enumerate(self.local_agents):
                    if j == agent_i:
                        continue
                    local_evaluator_estimates["values"].append(
                        evaluator.agent.evaluate_actions(
                            self.best_obs, action.detach(),
                        )
                    )

                for k, v in local_evaluator_estimates.items():
                    evaluator_estimates[k].append(v)

            # aggregate all value and log probability estimates across
            # all agents
            collective_values = self.aggregate_collective_estimates(
                proposer_estimates, evaluator_estimates, "values"
            )

            # rank candidates based on highest predicted value weighted by
            # the action probability
            weighted_values = (
                torch.stack(proposer_estimates["values"]) *
                torch.exp(torch.stack(proposer_estimates["log_probs"]))
            )

            best_idx = torch.argsort(
                weighted_values, descending=True
            )[0].item()

            self.collective_memory.record(
                agent_index[best_idx],
                *[v[best_idx] for _, v in proposer_estimates.items()] + [
                    collective_values[best_idx]
                ]
            )

            suggestions.append(candidates[best_idx])

        # experience replay: generate log_probs, entropy, values for all
        # past experiences.
        return suggestions

    def aggregate_collective_estimates(
        self, proposer_estimates, evaluator_estimates, key, weight=1.0
    ):
        if self.n_agents == 1:
            return torch.stack(proposer_estimates[key])

        return torch.cat(
            [
                torch.stack(proposer_estimates[key]).view(-1, 1) * weight,
                torch.stack([torch.stack(x) for x in evaluator_estimates[key]])
            ],
            dim=1
        ).mean(dim=1)

    def update(self, y, entropy_coef=0.0):
        # TODO: enable full-history learning by keeping track of all the
        # actions, observations, outcomes, and rewards. At each update step,
        # compute actor/critic losses for current time step in addition to
        # the past time steps.
        # - generate log prob and entropies estimate using current policy based
        #   on past actions
        # - generate value estimates and derive reward using global history

        # aggregate all y's (raw values) and derive reward
        self.n_updates += 1

        if self.n_updates <= 1:
            rewards = derive_reward(y, self.best_y)
            return self.init_update(y, rewards, entropy_coef)

        replay_results = self.experience_replay()

        global_y = y + replay_results["outcomes"]
        rewards = derive_reward(global_y)

        # compute actor critic losses based on current and past actions
        advantage = rewards - torch.stack(
            self.collective_memory.memory.values + replay_results["values"]
        )
        actor_loss = (
            torch.stack(
                self.collective_memory.memory.log_probs
                + replay_results["log_probs"]
            )
            * advantage
        )
        critic_loss = 0.5 * advantage ** 2

        # collective critic
        if self.n_agents > 1:
            collective_advantage = (
                rewards - torch.stack(self.collective_memory.collective_values)
            )
            collective_critic_loss = 0.5 * collective_advantage ** 2
        else:
            collective_critic_loss = torch.tensor(0).float()

        entropy_loss = (
            torch.stack(
                self.collective_memory.memory.entropies
                + replay_results["entropies"]
            ) * entropy_coef
        )
        loss = (
            actor_loss.mean()
            + critic_loss.mean()
            + collective_critic_loss.mean() * self.kappa
            - entropy_loss.mean()
        )

        agent_memories = []
        for local_agent in self.local_agents:
            agent_memories.append(local_agent.agent.memory.detach())

        # update reward, success and failure states
        self.update_states(global_y, rewards)
        self.history.record(
            self.collective_memory.detach(), y,
        )
        return loss

    def experience_replay(self):
        if len(self.history) == 0:
            return None

        replay_results = defaultdict(list)
        agent_experiences = defaultdict(lambda: defaultdict(list))
        for agent_index, obs, action, outcome in self.history.recall():
            agent_experiences[agent_index]["obs"].append(obs)
            agent_experiences[agent_index]["actions"].append(action)
            agent_experiences[agent_index]["outcomes"].append(outcome)

        for agent_index, local_agent in enumerate(self.local_agents):
            experiences = agent_experiences[agent_index]
            for obs, action, outcome in zip(
                experiences["obs"],
                experiences["actions"],
                experiences["outcomes"],
            ):
                value, log_probs, entropy = (
                    local_agent.agent.evaluate_experiences(
                        torch.from_numpy(obs).float(),
                        torch.from_numpy(action).float(),
                    )
                )
                replay_results["values"].append(value)
                replay_results["log_probs"].append(log_probs)
                replay_results["entropies"].append(entropy)
                replay_results["outcomes"].append(outcome)

        # TODO: assert that all entries in replay_results are same length
        return replay_results

    def update_states(self, y, rewards):
        best_reward = rewards.max()
        best_idx = np.argmax(rewards)
        best_y = y[best_idx]

        # assume lower scores are better
        update_best_obs = best_y < self.best_y

        if update_best_obs:
            print("[updating best observation]")
            self.best_reward = best_reward
            self.best_y = best_y
            self.best_obs = (
                self.collective_memory.memory.adjusted_obs[best_idx]
                .detach()
            )
            self.n_successes += 1
        else:
            self.n_failures += 1

        # update local agent states
        # allocate rewards and candidates to the corresponding agents
        agent_rewards = defaultdict(list)
        agent_candidates = defaultdict(list)
        for r, idx, candidate in zip(
            rewards,
            self.collective_memory.agent_index,
            self.collective_memory.memory.adjusted_obs,
        ):
            agent_rewards[idx].append(r)
            agent_candidates[idx].append(candidate.detach().numpy())

        # update agent states with their rewards
        for idx, r in agent_rewards.items():
            self.local_agents[idx].update_states(r, agent_candidates[idx])


class CollectiveMemory:
    def __init__(self, agent_index=None, memory=None):
        self.agent_index: List[
            int
        ] = [] if agent_index is None else agent_index
        self.memory: Memory = Memory() if memory is None else memory
        self.collective_values = []

    def record(
        self, agent_index, obs, action, adjusted_obs, log_probs, entropy,
        value, prob_dist, collective_values
    ):
        self.agent_index.append(agent_index)
        self.memory.obs.append(obs)
        self.memory.actions.append(action)
        self.memory.adjusted_obs.append(adjusted_obs)
        self.memory.log_probs.append(log_probs)
        self.memory.entropies.append(entropy)
        self.memory.values.append(value)
        self.memory.prob_dists.append(prob_dist)
        self.collective_values.append(collective_values)

    def erase(self):
        self.memory.erase()
        del self.agent_index[:]
        del self.collective_values[:]

    def detach(self):
        memory = CollectiveMemory(
            deepcopy(self.agent_index), self.memory.detach()
        )
        self.erase()
        return memory

    def __len__(self):
        return len(self.agent_index)


class History:
    def __init__(self):
        self.collective_memories: List[CollectiveMemory] = []
        self.outcomes: List[List[float]] = []

    def record(self, cmemory, outcomes):
        self.collective_memories.append(deepcopy(cmemory))
        self.outcomes.append(deepcopy(outcomes))

    def recall(self):
        """Iterate through memories with most recent ones first."""
        for cmemory, outcomes in zip(
            reversed(self.collective_memories), reversed(self.outcomes),
        ):
            for agent_index, obs, action, outcome in zip(
                reversed(cmemory.agent_index),
                reversed(cmemory.memory.obs),
                reversed(cmemory.memory.actions),
                reversed(outcomes),
            ):
                yield agent_index, obs, action, outcome

    def erase(self):
        del self.collective_memories[:]
        del self.outcomes[:]

    def __getitem__(self, i):
        return self.collective_memories[i], self.outcomes[i]

    def __len__(self):
        return len(self.collective_memories)


class LocalAgent:
    def __init__(self, agent):
        self.agent = agent

        # default attributes
        self.best_reward = float("-inf")
        self.best_suggestion = None
        self.n_successes = 0
        self.n_failures = 0

        # counters
        self.resample_counter = 0

    def create_candidates(self, obs, n, cov_scaler=1.0):
        self.agent.train()

        candidates = []
        out_actions = []
        for i in range(n):
            actions, adjusted_obs, log_probs, entropy, value, prob_dist = (
                self.agent(obs, cov_scaler)
            )
            candidates.append(adjusted_obs)
            out_actions.append(actions)

        return candidates, out_actions

    def update_states(
        self, rewards: List[float], candidates: List[np.ndarray]
    ):
        best_reward = max(rewards)
        best_idx = rewards.index(best_reward)
        if best_reward > self.best_reward:
            self.best_reward = best_reward
            self.best_suggestion = candidates[best_idx]
            self.n_successes += 1
        else:
            self.n_failures += 1

    def reset_states(self):
        self.n_successes = 0
        self.n_failures = 0
