from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn

from metalearn.algorithm_space import AlgorithmSpace
from metalearn.components import algorithm_component, hyperparameter
from metalearn.data_types import AlgorithmType
from metalearn.metalearn_controller import MetaLearnController
from metalearn.metalearn_reinforce import _check_buffers
from metalearn import scorers

from bayesmark.abstract_optimizer import AbstractOptimizer
from bayesmark.experiment import experiment_main
from bayesmark.space import JointSpace


EPSILON = np.finfo(np.float32).eps.item()
CLIP_GRAD = 20.0


class GridHyperparameter(hyperparameter.HyperparameterBase):
    """Hyperparameters specified as grid of choices."""

    def __init__(self, name, state_space):
        # setting default is required argument has no effect on state space
        super().__init__(name, state_space, default=state_space[0])


class Algorithm:
    """Generic object, needed by metalearn API but not really used."""
    def __init__(self, **kwargs):
        pass


def create_algorithm_space(api_config, max_interp):
    # use bayesmark JointSpace class to pre-define grid of values
    space = JointSpace(api_config)
    grid_space = {}
    for param, props in api_config.items():
        if props["type"] == "cat":
            grid_space[param] = props["values"]
        elif props["type"] == "bool":
            grid_space[param] = [0, 1]
        else:
            values = space.spaces[param].grid(max_interp)
            if props["type"] == "int":
                values = [np.int_(x) for x in values]
            elif props["type"] == "float":
                values = [np.float_(x) for x in values]

            # make sure min and max range are adhered to
            if values[0] < props["range"][0]:
                values[0] = props["range"][0]
            if values[1] > props["range"][1]:
                values[1] = props["range"][1]

            grid_space[param] = values

    hyperparameters = [
        GridHyperparameter(name, state_space) for name, state_space
        in grid_space.items()
    ]

    return AlgorithmSpace(
        classifiers=[
            algorithm_component.AlgorithmComponent(
                "Algorithm",
                Algorithm,
                component_type=AlgorithmType.ESTIMATOR,
                hyperparameters=hyperparameters,
            )
        ],
        regressors=[],
        data_preprocessors=[],
        feature_preprocessors=[],
        random_state=2001
    )


def create_controller(algorithm_space):
    return MetaLearnController(
        metafeature_size=1,
        input_size=50,
        hidden_size=50,
        output_size=50,
        a_space=algorithm_space,
        mlf_signature=[AlgorithmType.ESTIMATOR],
        dropout_rate=0.0,
        num_rnn_layers=1,
    )


def scalar_tensor_3d(val):
    x = torch.zeros(1, 1, 1)
    return x + val


class MetalearnOptimizer(AbstractOptimizer):
    primary_import = "meta-ml"

    def __init__(
        self,
        api_config,
        max_interp=30,
        gamma=0.99,
        entropy_coef=0.0,
        normalize_reward=False,
    ):
        """Build wrapper class to use optimizer in benchmark.

        Parameters
        ----------
        api_config : dict-like of dict-like
            Configuration of the optimization variables. See API description.
        """
        AbstractOptimizer.__init__(self, api_config)
        self.algorithm_space = create_algorithm_space(api_config, max_interp)
        self.controller = create_controller(self.algorithm_space)

        # optimizer
        self.optim = torch.optim.Adam(
            self.controller.parameters(),
            lr=0.0003,
            betas=(0.9, 0.999)
        )

        # hyperparameters
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.normalize_reward = normalize_reward

        # initial states
        self.last_value = None
        self.prev_reward = scalar_tensor_3d(0)
        self.prev_action = self.controller.init_action()
        self.prev_hidden = self.controller.init_hidden()

        # track controller states
        self.action_activations = []

        # track optimization stats
        self.call_metrics = defaultdict(int)
        self.history = defaultdict(list)

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
        self.call_metrics["suggest_calls"] += 1

        suggestions = []
        prev_action = self.prev_action
        prev_hidden = self.prev_hidden
        for i in range(n_suggestions):
            value, action, action_activation, hidden = self.controller(
                prev_action=prev_action,
                prev_reward=self.prev_reward,
                hidden=prev_hidden,
                # metafeatures don't apply so it'll just be a constant
                metafeatures=scalar_tensor_3d(1),
                target_type=None,
            )

            self.action_activations.append(action_activation)
            self.controller.value_buffer.append(value)
            self.controller.log_prob_buffer.append(
                [a["log_prob"] for a in action]
            )
            self.controller.entropy_buffer.append(
                [a["entropy"] for a in action]
            )

            suggestions.append(self.action_to_suggestion(action))
            prev_action, prev_hidden = action_activation, hidden

        # get last value for computing Q values
        self.last_value, *_ = self.controller(
            prev_action=prev_action,
            prev_reward=self.prev_reward,
            hidden=prev_hidden,
            metafeatures=scalar_tensor_3d(1),
            target_type=None,
        )

        self.prev_action = prev_action.detach()
        self.prev_hidden = prev_hidden.detach()

        return suggestions

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
        # For metrics where higher values are better, e.g. accuracy, y will
        # contain negative values. Negate negative values since the objective
        # value is a minimization problem, and we want a reward, where higher
        # is better.
        higher_is_better = any(x < 0 for x in y)

        # if lower scores are better, assume a metric where 0 is the best score
        # and inf is the worst, e.g. MSE, NLL. In these cases, bound scores so
        # that 0.0 maps to 1.0 and inf maps to 0
        rewards = [
            -x if higher_is_better else
            scorers.exponentiated_log(x)
            for x in y
        ]

        self.call_metrics["observe_calls"] += 1
        self.controller.reward_buffer.extend(rewards)
        # set previous reward to mean of rewards
        self.prev_reward = scalar_tensor_3d(np.mean(rewards))
        self.update_controller()

    def action_to_suggestion(self, action):
        """Convert controller action output into suggestion."""
        hyperparameters = {
            x["action_name"].replace("Algorithm__", ""): x["choice"]
            # the first action is always the estimator
            for x in action[1:]
        }

        assert hyperparameters.keys() == self.api_config.keys()
        return hyperparameters

    def update_controller(self):
        _check_buffers(
            self.controller.value_buffer,
            self.controller.log_prob_buffer,
            self.controller.reward_buffer,
            self.controller.entropy_buffer)

        n = len(self.controller.reward_buffer)

        # reset gradient
        self.optim.zero_grad()

        # compute Q values
        returns = torch.zeros(len(self.controller.value_buffer))
        R = self.last_value
        for t in reversed(range(n)):
            R = self.controller.reward_buffer[t] + self.gamma * R
            returns[t] = R

        # mean-center and std-scale returns
        if self.normalize_reward:
            returns = (returns - returns.mean()) / (returns.std() + EPSILON)

        values = torch.cat(self.controller.value_buffer).squeeze()
        advantage = returns - values

        # compute loss
        actor_loss = [
            -log_prog * action
            for log_probs, action in zip(
                self.controller.log_prob_buffer, advantage
            )
            for log_prog in log_probs
        ]
        actor_loss = torch.cat(actor_loss).mean()
        critic_loss = 0.5 * advantage.pow(2).mean()

        # entropy loss term, negate the mean since we want to maximize entropy
        entropies = torch.cat([
            e * self.entropy_coef
            for entropy_list in self.controller.entropy_buffer
            for e in entropy_list
        ]).squeeze()
        entropy_loss = -entropies.mean()

        actor_critic_loss = actor_loss + critic_loss + entropy_loss

        # one step of gradient descent
        actor_critic_loss.backward()
        # gradient clipping to prevent exploding gradient
        nn.utils.clip_grad_norm_(self.controller.parameters(), CLIP_GRAD)
        self.optim.step()

        grad_norm = 0.
        for p in self.controller.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                grad_norm += param_norm.item() ** 2
        grad_norm = grad_norm ** 0.5

        # reset rewards and log probs
        del self.controller.value_buffer[:]
        del self.controller.log_prob_buffer[:]
        del self.controller.reward_buffer[:]
        del self.controller.entropy_buffer[:]

        self.history["actor_critic_loss"].append(actor_critic_loss.data.item())
        self.history["actor_loss"].append(actor_loss.data.item())
        self.history["critic_loss"].append(critic_loss.data.item())
        self.history["entropy_loss"].append(entropy_loss.data.item())
        self.history["grad_norm"].append(grad_norm)


if __name__ == "__main__":
    # This is the entry point for experiments, so pass the class to
    # experiment_main to use this optimizer.
    # This statement must be included in the wrapper class file:
    experiment_main(MetalearnOptimizer)
