from typing import Union, Optional, Any, Dict, List, Tuple, Set
try:
    import wandb
except Exception:  # make wandb optional
    class _DummyWandb:
        def login(self, *args, **kwargs):
            pass
        def init(self, *args, **kwargs):
            pass
        def log(self, *args, **kwargs):
            pass
        def finish(self):
            pass
    wandb = _DummyWandb()
import os
import numpy as np
import jax
import jax.random as jr
import functools as ft

from time import time
from tqdm import tqdm

from .data import Rollout
from .utils import rollout
from ..env import MultiAgentEnv
from ..algo.base import MultiAgentController
from ..utils.utils import jax_vmap

def _is_safe_flags(graph_t):
    """Compute per-agent safety flags for a single time step graph.
    Safe := agent not overlapping or Chebyshev-adjacent (<=1) to any other agent or obstacle.
    Returns: (N,) boolean array"""
    import jax.numpy as jnp
    agent_pos = graph_t.env_states.agent
    N = agent_pos.shape[0]
    diffs = agent_pos[:, None, :] - agent_pos[None, :, :]
    man = jnp.abs(diffs[..., 0]) + jnp.abs(diffs[..., 1])
    unsafe_agent_agent = (man <= 1) & (~jnp.eye(N, dtype=bool))
    unsafe_from_agents = jnp.any(unsafe_agent_agent, axis=1)
    obs_pos = getattr(graph_t.env_states, 'obs', None)
    if obs_pos is None:
        unsafe_from_obstacles = jnp.zeros((N,), dtype=bool)
    else:
        diffs_obs = agent_pos[:, None, :] - obs_pos[None, :, :]
        man_obs = jnp.abs(diffs_obs[..., 0]) + jnp.abs(diffs_obs[..., 1])
        unsafe_from_obstacles = jnp.any(man_obs <= 1, axis=1)
    safe = ~(unsafe_from_agents | unsafe_from_obstacles)
    return safe

def _has_reached_flags(graph_t):
    """Per-agent reached-goal flags for a single time step graph.
    Reach := |pos - goal| < 1.0 elementwise (L_infty < 1.0)."""
    import jax.numpy as jnp
    agent_pos = graph_t.env_states.agent
    goal_pos = graph_t.env_states.goal
    diff = jnp.abs(agent_pos - goal_pos)
    # Manhattan distance == 0 (完全重合)
    reached = jnp.all(diff < 1e-6, axis=-1)
    return reached



class Trainer:

    def __init__(
            self,
            env: MultiAgentEnv,
            env_test: MultiAgentEnv,
            algo: MultiAgentController,
            n_env_train: int,
            n_env_test: int,
            log_dir: str,
            seed: int,
            params: dict,
            save_log: bool = True
    ):
        self.env = env
        self.env_test = env_test
        self.algo = algo
        self.n_env_train = n_env_train
        self.n_env_test = n_env_test
        self.log_dir = log_dir
        self.seed = seed

        if Trainer._check_params(params):
            self.params = params

        # make dir for the models
        if save_log:
            if not os.path.exists(log_dir):
                os.mkdir(log_dir)
            self.model_dir = os.path.join(log_dir, 'models')
            if not os.path.exists(self.model_dir):
                os.mkdir(self.model_dir)

        # 禁用wandb登录以避免API key问题
        try:
            wandb.login(anonymous="allow")
            wandb.init(name=params['run_name'], project='gcbf-jax', dir=self.log_dir, mode="disabled")
            print("✅ Wandb初始化成功（禁用模式）")
        except Exception as e:
            print(f"⚠️  Wandb初始化失败，继续训练: {e}")
            # 创建一个假的wandb对象
            class DummyWandb:
                def log(self, *args, **kwargs):
                    pass
                def finish(self):
                    pass
            import sys
            sys.modules['wandb'] = DummyWandb()

        self.save_log = save_log

        self.steps = params['training_steps']
        self.eval_interval = params['eval_interval']
        self.eval_epi = params['eval_epi']
        self.save_interval = params['save_interval']

        self.update_steps = 0
        self.key = jax.random.PRNGKey(seed)

    @staticmethod
    def _check_params(params: dict) -> bool:
        assert 'run_name' in params, 'run_name not found in params'
        assert 'training_steps' in params, 'training_steps not found in params'
        assert 'eval_interval' in params, 'eval_interval not found in params'
        assert params['eval_interval'] > 0, 'eval_interval must be positive'
        assert 'eval_epi' in params, 'eval_epi not found in params'
        assert params['eval_epi'] >= 1, 'eval_epi must be greater than or equal to 1'
        assert 'save_interval' in params, 'save_interval not found in params'
        assert params['save_interval'] > 0, 'save_interval must be positive'
        return True

    def train(self):
        # record start time
        start_time = time()

        # preprocess the rollout function
        def rollout_fn_single(params, key):
            return rollout(self.env, ft.partial(self.algo.step, params=params), key)

        def rollout_fn(params, keys):
            return jax.vmap(ft.partial(rollout_fn_single, params))(keys)

        rollout_fn = jax.jit(rollout_fn)

        # preprocess the test function
        def test_fn_single(params, key):
            return rollout(self.env_test, lambda graph, k: (self.algo.act(graph, params), None), key)

        def test_fn(params, keys):
            return jax.vmap(ft.partial(test_fn_single, params))(keys)

        test_fn = jax.jit(test_fn)

        # start training
        test_key = jr.PRNGKey(self.seed)
        test_keys = jr.split(test_key, 1_000)[:self.n_env_test]

        pbar = tqdm(total=self.steps, ncols=80)
        for step in range(0, self.steps + 1):
            # evaluate the algorithm
            if step % self.eval_interval == 0:
                test_rollouts: Rollout = test_fn(self.algo.actor_params, test_keys)
                total_reward = test_rollouts.rewards.sum(axis=-1)
                assert total_reward.shape == (self.n_env_test,)
                reward_min, reward_max = total_reward.min(), total_reward.max()
                reward_mean = np.mean(total_reward)
                # === Safety & Arrival metrics (GCBF+ definitions, re-implemented here) ===
                _vmap_env_time_safe = jax_vmap(jax_vmap(_is_safe_flags))
                _vmap_env_time_reach = jax_vmap(jax_vmap(_has_reached_flags))
                safe_mask = _vmap_env_time_safe(test_rollouts.graph)
                reach_mask = _vmap_env_time_reach(test_rollouts.graph)
                safety_ratio = np.array(safe_mask).mean()
                arrival_rate = np.array(reach_mask).max(axis=1).mean()
                finish_fun = jax_vmap(jax_vmap(self.env_test.finish_mask))
                finish = finish_fun(test_rollouts.graph).max(axis=1).mean()
                cost = test_rollouts.costs.sum(axis=-1).mean()
                unsafe_frac = np.mean(test_rollouts.costs.max(axis=-1) >= 1e-6)
                eval_info = {
                    "eval/reward": reward_mean,
                    "eval/cost": cost,
                    "eval/unsafe_frac": unsafe_frac,
                    "eval/safety_ratio": float(safety_ratio),
                    "eval/arrival_rate": float(arrival_rate),
                    "step": step,
                }
                wandb.log(eval_info, step=self.update_steps)
                try:
                    os.makedirs(self.log_dir, exist_ok=True)
                    with open(os.path.join(self.log_dir, 'eval_metrics.jsonl'), 'a') as f:
                        import json as _json
                        f.write(_json.dumps({**eval_info, 'update_step': int(self.update_steps)}) + '\n')
                except Exception as _e:
                    tqdm.write(f'[warn] failed to write eval_metrics.jsonl: {_e}')
                time_since_start = time() - start_time
                eval_verbose = (f'step: {step:3}, time: {time_since_start:5.0f}s, reward: {reward_mean:9.4f}, '
                                f'min/max reward: {reward_min:7.2f}/{reward_max:7.2f}, cost: {cost:8.4f}, '
                                f'unsafe_frac: {unsafe_frac:6.2f}, safety: {safety_ratio:6.2f}, arrival: {arrival_rate:6.2f}')
                tqdm.write(eval_verbose)
                if self.save_log and step % self.save_interval == 0:
                    self.algo.save(os.path.join(self.model_dir), step)

            # collect rollouts
            key_x0, self.key = jax.random.split(self.key)
            key_x0 = jax.random.split(key_x0, self.n_env_train)
            rollouts: Rollout = rollout_fn(self.algo.actor_params, key_x0)

            # update the algorithm
            update_info = self.algo.update(rollouts, step)
            wandb.log(update_info, step=self.update_steps)
            self.update_steps += 1

            pbar.update(1)
