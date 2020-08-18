import copy
import hashlib
import os
from collections import defaultdict
from pathlib import Path

import dill
import numpy as np
import torch
import torch.nn as nn
import scipy.stats as ss

from metalearn.algorithm_space import AlgorithmSpace
from metalearn.components import algorithm_component, hyperparameter
from metalearn.data_types import AlgorithmType
from metalearn.metalearn_controller import MetaLearnController
from metalearn.metalearn_reinforce import _check_buffers
from bayesmark.abstract_optimizer import AbstractOptimizer
from bayesmark.experiment import experiment_main


EPSILON = np.finfo(np.float32).eps.item()
CLIP_GRAD = 1.0


class Algorithm:
    """Generic object, needed by metalearn API but not really used."""
    def __init__(self, **kwargs):
        pass


def hash_api_config(api_config):
    algo_hash = hashlib.sha256()
    algo_hash.update(str(api_config).encode())
    return algo_hash.hexdigest()[:7]


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


def create_algorithm_space(api_config):
    # use bayesmark JointSpace class to pre-define grid of values
    grid_space = defaultdict(dict)

    hyperparameters = []

    for param, props in api_config.items():
        grid_space[param]["type"] = props["type"]
        if props["type"] == "cat":
            hyperparameters.append(
                hyperparameter.CategoricalHyperparameter(
                    param, props["values"], default=None
                )
            )
        elif props["type"] == "bool":
            hyperparameters.append(
                hyperparameter.CategoricalHyperparameter(
                    param, [0, 1], default=None
                )
            )
        else:
            if props["type"] == "int":
                hyperparam_cls = hyperparameter.UniformIntHyperparameter
            elif props["type"] == "real":
                hyperparam_cls = hyperparameter.UniformFloatHyperparameter
            else:
                raise ValueError(f"type not recognized: {props['type']}")

            hyperparameters.append(
                hyperparam_cls(
                    param, props["range"][0], props["range"][1]
                )
            )

    return AlgorithmSpace(
        classifiers=[
            algorithm_component.AlgorithmComponent(
                f"Algorithm_{hash_api_config(api_config)}",
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


def create_controller(
    algorithm_space,
    model_size=64,
    num_rnn_layers=3,
    dropout_rate=0.0,
    **kwargs
):
    return MetaLearnController(
        metafeature_size=1,
        input_size=model_size,
        hidden_size=model_size,
        output_size=model_size,
        a_space=algorithm_space,
        mlf_signature=[AlgorithmType.ESTIMATOR],
        dropout_rate=dropout_rate,
        num_rnn_layers=num_rnn_layers,
        **kwargs
    )


def scalar_tensor_3d(val):
    x = torch.zeros(1, 1, 1).requires_grad_()
    return x + val


def load_pretrained_metalearner(path, algorithm_space, **kwargs):
    pretrained_controller = torch.load(path, pickle_module=dill)
    return create_controller(
        algorithm_space,
        pretrained_controller=pretrained_controller,
        **kwargs,
    )


class MetalearnOptimizer(AbstractOptimizer):
    primary_import = "meta-ml"
    pretrained_dir = Path(os.path.dirname(__file__)) / "pretrained"

    def __init__(
        self,
        api_config,
        pretrained_model_name=None,
        model_name=None,
        model_size=64,
        num_rnn_layers=3,
        learning_rate=0.03,
        weight_decay=0.1,
        dropout_rate=0.1,
        gamma=0.0,
        entropy_coef=100.0,
        entropy_factor=0.9,
        normalize_reward=True,
        eps_clip=1.0,
    ):
        """Build wrapper class to use optimizer in benchmark.

        Parameters
        ----------
        api_config : dict-like of dict-like
            Configuration of the optimization variables. See API description.
        """
        AbstractOptimizer.__init__(self, api_config)

        # TODO: [idea] the api_config string representation can be vectorized
        # using a char-level RNN/CNN, which could be used as the auxiliary
        # input environment state

        self.pretrained_model_name = pretrained_model_name
        self.model_name = model_name
        self.model_file = None

        if self.model_name is not None:
            model_dir = self.pretrained_dir / self.model_name
            model_dir.mkdir(exist_ok=True, parents=True)
            self.model_file = model_dir / "model.pickle"

        if self.model_file is not None and self.model_file.exists():
            # read in model if it exists already
            print("loading existing model in this run %s" % self.model_file)
            self.controller = load_pretrained_metalearner(
                self.pretrained_dir / model_name / "model.pickle",
                create_algorithm_space(api_config),
                model_size=model_size,
                num_rnn_layers=num_rnn_layers,
                dropout_rate=dropout_rate,
            )
        elif self.pretrained_model_name is not None:
            print("loading pre-trained model %s" % self.pretrained_model_name)
            self.controller = load_pretrained_metalearner(
                (
                    self.pretrained_dir / self.pretrained_model_name /
                    "model.pickle"
                ),
                create_algorithm_space(api_config),
                model_size=model_size,
                num_rnn_layers=num_rnn_layers,
                dropout_rate=dropout_rate,
            )
        else:
            print("initializing new model")
            self.controller = create_controller(
                create_algorithm_space(api_config),
                model_size=model_size,
                num_rnn_layers=num_rnn_layers,
                dropout_rate=dropout_rate
            )

        self.n_candidates = min(len(api_config) * 50, 5000)

        # optimizer
        self.optim = torch.optim.Adam(
            self.controller.parameters(),
            lr=learning_rate,
            betas=(0.9, 0.999),
            weight_decay=weight_decay,
        )

        self.memory = {"action_logprobs": []}

        # hyperparameters
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.dropout_rate = dropout_rate
        self.entropy_factor = entropy_factor
        self.eps_clip = eps_clip

        # initial states
        self.last_value = None
        self.prev_action = self.controller.init_action()
        self.prev_hidden = self.controller.init_hidden()
        self.prev_reward = scalar_tensor_3d(0)
        self.global_reward_max = None

        # track optimization stats
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

        suggestions = []
        prev_action = self.prev_action
        prev_hidden = self.prev_hidden

        candidate_suggestions = []
        candidate_buffers = defaultdict(list)
        for i in range(self.n_candidates):
        # for _ in range(n_suggestions):
            value, action, action_activation, hidden = self.controller(
                # prev_action=prev_action,
                # prev_reward=self.prev_reward,
                # hidden=prev_hidden,
                prev_action=self.controller.init_action(),
                prev_reward=scalar_tensor_3d(1),
                hidden=self.controller.init_hidden(),
                # metafeatures don't apply so it'll just be a constant
                metafeatures=scalar_tensor_3d(1),
                target_type=None,
            )

            candidate_buffers["values"].append(value.squeeze())
            # exclude algorithm selection action
            candidate_buffers["log_prob"].append(
                torch.cat([a["log_prob"] for a in action[1:]])
            )
            candidate_buffers["entropy"].append(
                torch.cat([a["entropy"] for a in action[1:]])
            )

            candidate_suggestions.append(self.action_to_suggestion(action))
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

        # TODO: try generating more than n_suggestions, ranking by value
        # prediction, and sending off the top and bottom n_suggestions / 2
        # only add actions associated with the selected suggestions to the
        # {value, log_prob, entropy} buffers.

        suggestions = self.select_candidates(
            n_suggestions, candidate_suggestions, candidate_buffers
        )

        # make sure suggestions are within the bounds
        api_config = self.api_config
        for suggestion in suggestions:
            for param, val in suggestion.items():
                if api_config[param]["type"] in {"int", "real"}:
                    min_val, max_val = api_config[param]["range"]
                    assert min_val <= val <= max_val
                elif api_config[param]["cat"]:
                    assert val in api_config[param]["values"]
                elif api_config[param]["bool"]:
                    assert val in [0, 1]

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

        self.y = y  

        yy = copula_standardize(y)
        rewards = [x if higher_is_better else -x for x in yy]
        norm = (
            np.max(rewards)
            if self.global_reward_max is None
            else self.global_reward_max
        )
        norm_rewards = [r - 0 for r in rewards]
        self.global_reward_max = max(np.max(rewards), norm)
        print(
            "[reward dist]", {
                "best_y": np.max(y) if higher_is_better else np.min(y),
                "worst_y": np.min(y) if higher_is_better else np.max(y),
                "mean_y": np.mean(y),
                "mean_reward": np.mean(rewards),
                "std_reward": np.std(rewards),
                "max_reward": np.max(rewards),
            }
        )
        print("[max reward action]", X[np.argmax(rewards)])
        print("[min reward action]", X[np.argmin(rewards)])

        self.controller.reward_buffer.extend(norm_rewards)
        # set previous reward to mean of rewards
        self.prev_reward = scalar_tensor_3d(np.mean(norm_rewards))
        self.update_controller()

        # reset rewards and log probs
        del self.controller.value_buffer[:]
        del self.controller.log_prob_buffer[:]
        del self.controller.reward_buffer[:]
        del self.controller.entropy_buffer[:]

        # save the model after every update
        if self.model_name is not None:
            torch.save(
                self.controller,
                self.pretrained_dir / self.model_name / "model.pickle",
                pickle_module=dill
            )

        print(
            "[controller losses]",
            {
                k: self.history[k][-1] for k in [
                    "actor_critic_loss",
                    "actor_loss",
                    "critic_loss",
                    "entropy_loss",
                    "grad_norm",
                ]
            },
            "\n"
        )
        import time; time.sleep(3)

        # decrement entropy coef
        if self.entropy_coef > 0:
            self.entropy_coef *= self.entropy_factor
        if self.entropy_coef < 0:
            self.entropy_coef = 0

    def action_to_suggestion(self, action):
        """Convert controller action output into suggestion."""
        hyperparameters = {
            x["action_name"].split("__")[-1]: x["choice"]
            # the first action is always the estimator
            for x in action[1:]
        }

        assert hyperparameters.keys() == self.api_config.keys()
        return hyperparameters

    def select_candidates(
        self, n_suggestions, candidate_suggestions, candidate_buffers
    ):
        predicted_values = torch.stack(candidate_buffers["values"]).detach()
        top_idx = torch.argsort(predicted_values, descending=True)
        suggestions = []
        for i in top_idx[:n_suggestions]:
            suggestions.append(candidate_suggestions[i])
            self.controller.value_buffer.append(candidate_buffers["values"][i])
            self.controller.log_prob_buffer.append(
                candidate_buffers["log_prob"][i]
            )
            self.controller.entropy_buffer.append(
                candidate_buffers["entropy"][i]
            )
        return suggestions

    def update_controller(self):
        _check_buffers(
            self.controller.value_buffer,
            self.controller.log_prob_buffer,
            self.controller.reward_buffer,
            self.controller.entropy_buffer
        )

        n = len(self.controller.reward_buffer)

        # reset gradient
        self.optim.zero_grad()

        # compute Q values
        returns = torch.zeros(len(self.controller.value_buffer))
        next_value = 0
        for t in reversed(range(n)):
            R = self.controller.reward_buffer[t] + self.gamma * next_value
            returns[t] = R
            next_value = self.controller.value_buffer[t]

        returns = (returns - returns.mean()) / (returns.std() + 1e-5)
        values = torch.stack(self.controller.value_buffer)
        advantage = returns - values

        old_action_logprobs = (
            [torch.zeros_like(x) for x in self.controller.log_prob_buffer]
            if len(self.memory["action_logprobs"]) == 0
            else self.memory["action_logprobs"]
        )

        # compute the ratio of new policy actions / old policy actions.
        # this is nested because each reward/value is associated with multiple
        # micro-actions generated by the controller
        ratios = [
            torch.exp(x - y) for x, y in zip(
                self.controller.log_prob_buffer,
                old_action_logprobs,
            )
        ]

        # compute surrogate losses
        surr1 = torch.stack([
            r * adv
            for ratio, adv in zip(ratios, advantage)
            for r in ratio
        ])

        surr2 = torch.stack([
            r * adv
            for ratio, adv in zip(ratios, advantage)
            for r in (
                torch.clamp(ratio, 1.0 - self.eps_clip, 1.0 + self.eps_clip)
            )
        ])

        # print("y", self.y)
        # print("global max reward", self.global_reward_max)
        # print("rewards", self.controller.reward_buffer)
        # print("returns", returns)
        # print("values", values)
        # print("advantage", advantage)
        # import ipdb; ipdb.set_trace()

        actor_loss = -torch.min(surr1, surr2).mean()
        critic_loss = 0.5 * advantage.pow(2).mean()

        # entropy loss term, negate the mean since we want to maximize entropy
        entropy_loss = -torch.mean(
            torch.cat(self.controller.entropy_buffer) * self.entropy_coef
        )
        actor_critic_loss = actor_loss + critic_loss + entropy_loss

        # one step of gradient descent
        actor_critic_loss.backward()
        # gradient clipping to prevent exploding gradient
        nn.utils.clip_grad_norm_(self.controller.parameters(), CLIP_GRAD)

        grad_norm = 0.
        for name, params in self.controller.named_parameters():
            if params.grad is not None:
                param_norm = params.grad.data.norm(2)
                grad_norm += param_norm.item() ** 2

        grad_norm = grad_norm ** 0.5
        self.optim.step()

        self.memory["action_logprobs"] = [
            x.detach() for x in self.controller.log_prob_buffer
        ]

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
