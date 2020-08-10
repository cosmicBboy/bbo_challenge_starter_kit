import os
from collections import defaultdict
from pathlib import Path

import dill
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

from pprint import pprint


EPSILON = np.finfo(np.float32).eps.item()
CLIP_GRAD = 1.0


class Algorithm:
    """Generic object, needed by metalearn API but not really used."""
    def __init__(self, **kwargs):
        pass


def create_algorithm_space(api_config, max_interp):
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
        input_size=64,
        hidden_size=64,
        output_size=64,
        a_space=algorithm_space,
        mlf_signature=[AlgorithmType.ESTIMATOR],
        dropout_rate=0.0,
        num_rnn_layers=3,
    )


def scalar_tensor_3d(val):
    x = torch.zeros(1, 1, 1)
    return x + val


def load_pretrained_metalearner(path, algorithm_space):
    model_config = torch.load(path, pickle_module=dill)
    # create a new controller based on algorithm_space
    controller = create_controller(algorithm_space)

    pretrained_algo, *_ = model_config["config"]["a_space"].classifiers
    algo, *_ = algorithm_space.classifiers

    # TODO: make sure that algorithm components are handled correctly here:
    # - hash the str repr of the api_config to uniquely identify api_configs
    # - loading a pretrained model should also load the pretrained "action_*"
    #   weights so that they are preserved
    if algo.hyperparameters != pretrained_algo.hyperparameters:
        # if pretrained algorithm has different hyperaparemeters, only
        # load meta_rnn, critic, and micro_action weights, ignoreing any
        # weights named "action_*"
        print("only loading meta-learning pre-trained weights")
        import ipdb; ipdb.set_trace()
        controller.load_state_dict({
            **{
                k: v for k, v in model_config["weights"].items()
                if not k.startswith("action_")
            },
            **{
                k: v for k, v in controller.named_parameters()
                if k.startswith("action_")
            }
        })
    else:
        print("loading all pre-trained weights")
        controller.load_state_dict(model_config["weights"])

    return controller


class MetalearnOptimizer(AbstractOptimizer):
    primary_import = "meta-ml"
    pretrained_dir = Path(os.path.dirname(__file__)) / "pretrained"

    def __init__(
        self,
        api_config,
        pretrained_model_name=None,
        model_name="submission_20200809_01",
        max_interp=10,
        learning_rate=0.03,
        weight_decay=0.1,
        dropout_rate=0.1,
        gamma=0.0,
        entropy_coef=1.0,
        entropy_decrement=0.0,
        normalize_reward=True,
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
                create_algorithm_space(api_config, max_interp)
            )
        if self.pretrained_model_name is not None:
            print("loading pre-trained model %s" % self.pretrained_model_name)
            self.controller = load_pretrained_metalearner(
                (
                    self.pretrained_dir / self.pretrained_model_name /
                    "model.pickle"
                ),
                create_algorithm_space(api_config, max_interp)
            )
        else:
            print("initializing new model")
            self.controller = create_controller(
                create_algorithm_space(api_config, max_interp)
            )

        # optimizer
        self.optim = torch.optim.Adam(
            self.controller.parameters(),
            lr=learning_rate,
            betas=(0.9, 0.999),
            weight_decay=weight_decay,
        )

        # hyperparameters
        self.learning_rate = learning_rate
        self.gamma = gamma
        self._entropy_coef = entropy_coef
        self.entropy_coef = entropy_coef
        self.entropy_decrement = entropy_decrement
        self.normalize_reward = normalize_reward

        # initial states
        self.last_value = None
        self.prev_reward = scalar_tensor_3d(0)

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
        prev_action = self.controller.init_action()
        prev_hidden = self.controller.init_hidden()
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

        # make sure suggestions are within the bounds
        # _check_suggestions(suggestions, self.api_config)
        api_config = self.api_config
        for suggestion in suggestions:
            for param, val in suggestion.items():
                if api_config[param]["type"] in {"int", "real"}:
                    min_val, max_val = api_config[param]["range"]
                    if not min_val <= val <= max_val:
                        import ipdb; ipdb.set_trace()
                elif api_config[param]["cat"]:
                    if val not in api_config[param]["values"]:
                        import ipdb; ipdb.set_trace()
                elif api_config[param]["bool"]:
                    if val not in [0, 1]:
                        import ipdb; ipdb.set_trace()

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
        print(
            "mean_reward: {:04f}; mean_y: {:04f}; max_reward: {:04f}".format(
                np.mean(rewards), np.mean(y), np.max(rewards)
            )
        )

        self.call_metrics["observe_calls"] += 1
        self.controller.reward_buffer.extend(rewards)
        # set previous reward to mean of rewards
        self.prev_reward = scalar_tensor_3d(np.mean(rewards))
        self.update_controller()

        # save the model after every update
        if self.model_name is not None:
            self.controller.save(
                self.pretrained_dir / self.model_name / "model.pickle"
            )

        # reset rewards and log probs
        del self.controller.value_buffer[:]
        del self.controller.log_prob_buffer[:]
        del self.controller.reward_buffer[:]
        del self.controller.entropy_buffer[:]

        print(
            {
                k: self.history[k][-1] for k in [
                    "actor_critic_loss",
                    "actor_loss",
                    "critic_loss",
                    "entropy_loss",
                    "grad_norm",
                ]
            }
        )

        # decrement entropy coef
        if self.entropy_coef > 0:
            self.entropy_coef -= self.entropy_decrement
        if self.entropy_coef < 0:
            self.entropy_coef = 0

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
        # _check_buffers(
        #     self.controller.value_buffer,
        #     self.controller.log_prob_buffer,
        #     self.controller.reward_buffer,
        #     self.controller.entropy_buffer)

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
            returns = returns - returns.mean()
            # sd = returns.std()
            # if sd == 0:
            #     returns = returns / (sd + EPSILON)

        values = torch.cat(self.controller.value_buffer).squeeze()
        advantage = returns - values

        # compute loss
        actor_loss = [
            -log_prob * action
            for log_probs, action in zip(
                self.controller.log_prob_buffer, advantage
            )
            for log_prob in log_probs
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

        grad_norm = 0.
        for name, params in self.controller.named_parameters():
            if params.grad is not None:
                param_norm = params.grad.data.norm(2)
                grad_norm += param_norm.item() ** 2

        grad_norm = grad_norm ** 0.5

        # if np.isnan(grad_norm):
        #     import ipdb; ipdb.set_trace()

        self.optim.step()

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
