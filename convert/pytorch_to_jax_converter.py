
import os
import pickle
from typing import Any, Dict, Optional

import jax.numpy as jnp
import flax.linen as nn
from flax.core import FrozenDict
import numpy as np

# Optional: only needed when converting from PyTorch -> JAX
try:
    import torch
except Exception:
    torch = None


# ================================
# JAX/Flax implementation of GAT-MF
# ================================

class JAXActor(nn.Module):
    """Flax MLP for the actor head used in GAT-MF.

    Input shape: (batch, n_agents, 10)  where 10 = concat([state(5), attn_agg_state(5)])
    Output: logits over 5 discrete actions, shape (batch, n_agents, 5)
    """
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(32, name="Dense_0")(x)  # Actor_hidden_features1
        x = nn.leaky_relu(x)
        x = nn.Dense(16, name="Dense_1")(x)  # Actor_hidden_features2
        x = nn.leaky_relu(x)
        x = nn.Dense(5, name="Dense_2")(x)   # 5 discrete actions
        return x


class JAXAttention(nn.Module):
    """Flax attention used by the actor / critic in the original PyTorch GAT-MF.
    We reproduce the exact computation: Q = s @ Wq, K = s @ Wk, att = square(Q @ K^T) * G
    then row-normalize over the last dim.
    """
    @nn.compact
    def __call__(self, s, Gmat):
        # s: (batch, n_agents, 5)
        # Gmat: (batch, n_agents, n_agents) or (n_agents, n_agents)
        if Gmat.ndim == 2:
            Gmat = jnp.expand_dims(Gmat, 0)  # (1, n_agents, n_agents)

        # Parameters (match PyTorch shapes [in, out] = (5, 32))
        Qweight = self.param('Qweight', nn.initializers.uniform(scale=0.5), (5, 32))
        Kweight = self.param('Kweight', nn.initializers.uniform(scale=0.5), (5, 32))

        q = jnp.einsum('ijk,km->ijm', s, Qweight)  # (batch, n_agents, 32)
        k = jnp.einsum('ijk,km->ijm', s, Kweight)  # (batch, n_agents, 32)
        k = jnp.transpose(k, (0, 2, 1))            # (batch, 32, n_agents)

        att = jnp.square(jnp.matmul(q, k)) * Gmat  # (batch, n_agents, n_agents)
        att = att / (jnp.sum(att, axis=2, keepdims=True) + 1e-3)
        return att


class GAT_MF_JAX:
    """JAX/Flax forward of GAT-MF used to produce u_ref for GCBF(+)."""

    def __init__(self, n_agents: int):
        self.n_agents = n_agents
        self.actor = JAXActor()
        self.actor_attention = JAXAttention()

    def __call__(self, params: Dict[str, Any], graph) -> jnp.ndarray:
        """Compute u_ref for a single graph.

        Args:
            params: {'actor': FrozenDict, 'actor_attention': FrozenDict}
            graph:  gcbf_plus.utils.graph.GraphsTuple or env GraphsTuple

        Returns:
            u_ref: (n_agents, 2) continuous control in [-1, 1]
        """
        # Extract agent / goal states (x, y)
        agent_states = graph.type_states(0, self.n_agents)  # (n_agents, 2)
        goal_states  = graph.type_states(1, self.n_agents)  # (n_agents, 2)

        # Features: [dx, dy, |dx|, |dy|, 1]
        dxdy = goal_states - agent_states
        dx, dy = dxdy[:, 0:1], dxdy[:, 1:2]
        ones = jnp.ones_like(dx)
        state_features = jnp.concatenate([dx, dy, jnp.abs(dx), jnp.abs(dy), ones], axis=1)  # (n_agents, 5)

        # Add batch dim
        state_features = jnp.expand_dims(state_features, 0)  # (1, n_agents, 5)

        # Fully-connected adjacency without self-loops (matches ref trainer defaults)
        Gmat = jnp.ones((self.n_agents, self.n_agents), dtype=state_features.dtype) - jnp.eye(self.n_agents, dtype=state_features.dtype)

        # Attention
        att = self.actor_attention.apply(params['actor_attention'], state_features, Gmat)  # (1, n_agents, n_agents)

        # Aggregate neighbor info
        att_agg = jnp.matmul(att, state_features)  # (1, n_agents, 5)

        # Actor forward on concatenated features
        actor_in = jnp.concatenate([state_features, att_agg], axis=-1)  # (1, n_agents, 10)
        logits = self.actor.apply(params['actor'], actor_in)            # (1, n_agents, 5)

        # Discrete -> continuous expectation
        probs = nn.softmax(logits, axis=-1)  # (1, n_agents, 5)
        action_vectors = jnp.array([
            [0.0,  0.0],   # stay
            [0.0,  1.0],   # up
            [0.0, -1.0],   # down
            [-1.0, 0.0],   # left
            [1.0,  0.0],   # right
        ], dtype=probs.dtype)  # (5,2)

        u_ref = jnp.einsum('ijk,kl->ijl', probs, action_vectors)  # (1, n_agents, 2)
        u_ref = u_ref.squeeze(0)                                   # (n_agents, 2)
        return jnp.clip(u_ref, -1.0, 1.0)


# ================================
# PyTorch -> Flax parameter converter
# ================================

def _assert_torch_available():
    if torch is None:
        raise RuntimeError("PyTorch is not available in this environment. Install torch to run conversion.")


def _to_flax_dense(pt_linear) -> Dict[str, np.ndarray]:
    """Map torch.nn.Linear (weight[out,in], bias[out]) -> Flax Dense (kernel[in,out], bias[out])."""
    W = pt_linear.weight.detach().cpu().numpy().T  # (in, out)
    b = pt_linear.bias.detach().cpu().numpy()
    return {"kernel": W, "bias": b}


class PyTorchToJAXConverter:
    """Convert trained PyTorch GAT-MF (Actor + Attention) to Flax parameter trees.
    Expected PyTorch modules are from grid_networks.py: Actor / Attention.
    """

    def convert_pytorch_to_jax(
        self,
        pytorch_actor_path: str,
        pytorch_attention_path: str,
        n_agents: int,
        save_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        _assert_torch_available()
        import sys, os
        sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
        # 1) Build PyTorch modules and load weights
        from gat_mf import grid_networks as networks  # local module in the same project
        actor_pt = networks.Actor()
        attn_pt = networks.Attention()
        actor_pt.load_state_dict(torch.load(pytorch_actor_path, map_location="cpu"))
        attn_pt.load_state_dict(torch.load(pytorch_attention_path, map_location="cpu"))
        actor_pt.eval(); attn_pt.eval()

        # 2) Map to Flax parameter trees
        actor_params_tree = {
            "params": {
                # Flax Dense names correspond to creation order: Dense_0, Dense_1, Dense_2
                "Dense_0": _to_flax_dense(actor_pt.lin1),
                "Dense_1": _to_flax_dense(actor_pt.lin2),
                "Dense_2": _to_flax_dense(actor_pt.lin3),
            }
        }
        attn_params_tree = {
            "params": {
                # Same shapes, direct copy
                "Qweight": attn_pt.Qweight.detach().cpu().numpy(),
                "Kweight": attn_pt.Kweight.detach().cpu().numpy(),
            }
        }

        flax_params: Dict[str, Any] = {
            "actor": FrozenDict(actor_params_tree),
            "actor_attention": FrozenDict(attn_params_tree),
            "n_agents": int(n_agents),
        }

        if save_path is not None:
            with open(save_path, "wb") as f:
                pickle.dump(flax_params, f)
        return flax_params


# ================================
# Helper: load/save + u_ref injection
# ================================

def save_flax_params(params: Dict[str, Any], path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(params, f)


def load_flax_params(path: str) -> Dict[str, Any]:
    with open(path, "rb") as f:
        return pickle.load(f)


def make_uref_from_flax_params(flax_params: Dict[str, Any]):
    """Return a callable(graph)->u_ref using the provided Flax params."""
    n_agents = int(flax_params["n_agents"]) if "n_agents" in flax_params else None
    if n_agents is None:
        raise ValueError("'n_agents' not found in flax_params.")
    gat = GAT_MF_JAX(n_agents=n_agents)
    params = {"actor": flax_params["actor"], "actor_attention": flax_params["actor_attention"]}

    def u_ref_fn(graph) -> jnp.ndarray:
        return gat(params, graph)

    return u_ref_fn


def inject_uref_to_env(env, flax_params: Dict[str, Any]) -> None:
    """env.set_u_ref_model(fn) if the env supports it (UnifiedGridEnv)."""
    if not hasattr(env, "set_u_ref_model") or not callable(getattr(env, "set_u_ref_model")):
        raise AttributeError("Environment does not expose set_u_ref_model(fn)." )
    env.set_u_ref_model(make_uref_from_flax_params(flax_params))


# ================================
# Minimal example (documentation)
# ================================
EXAMPLE_USAGE = """
# 1) Convert PyTorch weights -> Flax params (run once after PyTorch training)
from pytorch_to_jax_converter import PyTorchToJAXConverter, save_flax_params
converter = PyTorchToJAXConverter()
flax_params = converter.convert_pytorch_to_jax(
    pytorch_actor_path="/path/to/actor_xxx.pth",
    pytorch_attention_path="/path/to/actor_attention_xxx.pth",
    n_agents=4,
)
save_flax_params(flax_params, "/tmp/gat_mf_flax.pkl")

# 2) In JAX training script: load & inject into env before creating algo
from pytorch_to_jax_converter import load_flax_params, inject_uref_to_env
flax_params = load_flax_params("/tmp/gat_mf_flax.pkl")
inject_uref_to_env(env, flax_params)  # env.u_ref(graph) now calls GAT-MF forward

# 3) Create GCBF/GCBFPlus algo and Trainer as usual
"""
