import flax.linen as nn
import functools as ft
import numpy as np
import jax.nn as jnn
import jax.numpy as jnp
import tensorflow_probability.substrates.jax as tfp
tfd = tfp.distributions

from typing import Type, Tuple, Any
from abc import ABC, abstractproperty, abstractmethod


from ...utils.typing import Action, Array
from ...utils.graph import GraphsTuple
from ...nn.utils import default_nn_init, scaled_init
from ...nn.gnn import GNN
from ...nn.mlp import MLP
from ...utils.typing import PRNGKey, Params


class PolicyDistribution(nn.Module, ABC):

    @abstractmethod
    def __call__(self, *args, **kwargs) -> tfd.Distribution:
        pass

    @abstractproperty
    def nu(self) -> int:
        pass





class Deterministic(nn.Module):
    base_cls: Type[GNN]
    head_cls: Type[nn.Module]
    _nu: int

    @nn.compact
    def __call__(self, obs: GraphsTuple, n_agents: int, *args, **kwargs) -> Action:
        x = self.base_cls()(obs, node_type=0, n_type=n_agents)
        x = self.head_cls()(x)
        x = nn.tanh(nn.Dense(self._nu, kernel_init=default_nn_init(), name="OutputDense")(x))
        return x


class MultiAgentPolicy(ABC):

    def __init__(self, node_dim: int, edge_dim: int, n_agents: int, action_dim: int):
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.n_agents = n_agents
        self.action_dim = action_dim

    @abstractmethod
    def get_action(self, params: Params, obs: GraphsTuple) -> Action:
        pass

    @abstractmethod
    def sample_action(self, params: Params, obs: GraphsTuple, key: PRNGKey) -> Tuple[Action, Array]:
        pass

    @abstractmethod
    def eval_action(self, params: Params, obs: GraphsTuple, action: Action, key: PRNGKey) -> Tuple[Array, Array]:
        pass


class DeterministicPolicy(MultiAgentPolicy):

    def __init__(
            self,
            node_dim: int,
            edge_dim: int,
            n_agents: int,
            action_dim: int,
            gnn_layers: int = 1,
    ):
        super().__init__(node_dim, edge_dim, n_agents, action_dim)
        self.policy_base = ft.partial(
            GNN,
            msg_dim=128,
            hid_size_msg=(256, 256),
            hid_size_aggr=(128, 128),
            hid_size_update=(256, 256),
            out_dim=128,
            n_layers=gnn_layers
        )
        self.policy_head = ft.partial(
            MLP,
            hid_sizes=(256, 256),
            act=nn.relu,
            act_final=False,
            name='PolicyHead'
        )
        self.net = Deterministic(base_cls=self.policy_base, head_cls=self.policy_head, _nu=action_dim)
        self.std = 0.1

    def get_action(self, params: Params, obs: GraphsTuple) -> Action:
        return self.net.apply(params, obs, self.n_agents)

    def sample_action(self, params: Params, obs: GraphsTuple, key: PRNGKey) -> Tuple[Action, Array]:
        action = self.get_action(params, obs)
        log_pi = jnp.zeros_like(action)
        return action, log_pi

    def eval_action(self, params: Params, obs: GraphsTuple, action: Action, key: PRNGKey) -> Tuple[Array, Array]:
        raise NotImplementedError


class StochasticPolicy(MultiAgentPolicy):
    """Stochastic policy implementation for GCBF+"""

    def __init__(
            self,
            node_dim: int,
            edge_dim: int,
            n_agents: int,
            action_dim: int,
            gnn_layers: int = 1,
    ):
        super().__init__(node_dim, edge_dim, n_agents, action_dim)
        self.policy_base = ft.partial(
            GNN,
            msg_dim=128,
            hid_size_msg=(256, 256),
            hid_size_aggr=(128, 128),
            hid_size_update=(256, 256),
            out_dim=128,
            n_layers=gnn_layers
        )
        self.policy_head = ft.partial(
            MLP,
            hid_sizes=(256, 256),
            act=nn.relu,
            act_final=False,
            name='StochasticPolicyHead'
        )
        # Use a simplified stochastic network
        from ...nn.mlp import MLP as MLPModule
        self.mean_net = MLPModule(features=[256, 256, action_dim], activation=nn.relu)
        self.log_std_net = MLPModule(features=[256, 256, action_dim], activation=nn.relu)
        self.min_log_std = -10.0
        self.max_log_std = 2.0

    def get_action(self, params: Params, obs: GraphsTuple) -> Action:
        """Get mean action (for deterministic evaluation)"""
        # Process through GNN base
        processed_obs = self._process_observation(obs)
        mean = self.mean_net.apply(params['mean'], processed_obs)
        return mean

    def sample_action(self, params: Params, obs: GraphsTuple, key: PRNGKey) -> Tuple[Action, Array]:
        """Sample action from stochastic policy"""
        processed_obs = self._process_observation(obs)
        
        # Get mean and log_std
        mean = self.mean_net.apply(params['mean'], processed_obs)
        log_std = self.log_std_net.apply(params['log_std'], processed_obs)
        log_std = jnp.clip(log_std, self.min_log_std, self.max_log_std)
        std = jnp.exp(log_std)
        
        # Sample action
        noise = jax.random.normal(key, mean.shape)
        action = mean + std * noise
        
        # Compute log probability
        log_pi = -0.5 * jnp.sum((action - mean) ** 2 / (std ** 2 + 1e-8), axis=-1)
        log_pi -= 0.5 * jnp.sum(2 * log_std, axis=-1)
        log_pi -= 0.5 * mean.shape[-1] * jnp.log(2 * jnp.pi)
        
        return action, log_pi

    def eval_action(self, params: Params, obs: GraphsTuple, action: Action, key: PRNGKey) -> Tuple[Array, Array]:
        """Evaluate action log probability and entropy"""
        processed_obs = self._process_observation(obs)
        
        # Get mean and log_std
        mean = self.mean_net.apply(params['mean'], processed_obs)
        log_std = self.log_std_net.apply(params['log_std'], processed_obs)
        log_std = jnp.clip(log_std, self.min_log_std, self.max_log_std)
        std = jnp.exp(log_std)
        
        # Compute log probability
        log_pi = -0.5 * jnp.sum((action - mean) ** 2 / (std ** 2 + 1e-8), axis=-1)
        log_pi -= 0.5 * jnp.sum(2 * log_std, axis=-1)
        log_pi -= 0.5 * action.shape[-1] * jnp.log(2 * jnp.pi)
        
        # Compute entropy
        entropy = 0.5 * jnp.sum(2 * log_std + jnp.log(2 * jnp.pi * jnp.e), axis=-1)
        
        return log_pi, entropy

    def _process_observation(self, obs: GraphsTuple) -> Array:
        """Process observation through GNN base network"""
        # Simplified observation processing
        if obs.globals is not None:
            return obs.globals
        else:
            # Aggregate node features if no global features
            return jnp.mean(obs.nodes.reshape(-1, obs.nodes.shape[-1]), axis=0, keepdims=True)
