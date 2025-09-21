# gcbfplus/env/base.py
# 基于你项目里的 gcbf_grid_env.UnifiedGridEnv 接口，定义一个抽象环境基类。
from __future__ import annotations
from typing import Any, Dict, Optional, Tuple, Callable

import jax.numpy as jnp

from gcbf_plus.utils.typing import Action, State, Array
from gcbf_plus.utils.graph import GraphsTuple
from typing import NamedTuple, List


class RolloutResult(NamedTuple):
    """
    存储环境rollout结果的数据结构
    用于可视化和分析
    """
    Tp1_graph: List[GraphsTuple]  # 每个时间步的图结构 T+1个
    T_cost: List[float]           # 每个时间步的成本 T个  
    T_reward: List[float]         # 每个时间步的奖励 T个
    T_done: List[bool]            # 每个时间步的终止标志 T个
    T_info: List[Dict[str, Any]]  # 每个时间步的额外信息 T个
    
    @property
    def episode_length(self) -> int:
        """获取episode长度"""
        return len(self.T_cost)
    
    @property 
    def total_reward(self) -> float:
        """获取总奖励"""
        return sum(self.T_reward)
    
    @property
    def total_cost(self) -> float:
        """获取总成本"""
        return sum(self.T_cost)


class MultiAgentEnv:
    """
    多智能体图环境的抽象基类。

    设计目标：与 gcbf_grid_env.UnifiedGridEnv 完全“对齐”的 API，
    让 GCBF / GCBFPlus 只依赖本基类即可工作。

    子类（例如 UnifiedGridEnv）必须实现：
      - reset(key) -> GraphsTuple
      - step(graph, action) -> (next_graph, reward, cost, done, info)
      - forward_graph(graph, action) -> GraphsTuple
      - add_edge_feats(graph, new_state) -> GraphsTuple
      - control_affine_dyn(agent_state) -> (f, g)
      - safe_mask(graph) / finish_mask(graph)
      - action_lim() / state_lim()
      - 以及属性：action_dim / node_dim / edge_dim / dt / num_agents

    提供可选：set_u_ref_model(fn) 注入外部参考控制；若未注入，子类应重写 u_ref。
    """

    # -------------------------
    # 必需的只读属性（子类需要实现）
    # -------------------------
    @property
    def action_dim(self) -> int:
        """每个智能体的连续动作维度（例如 2 表示 (ux, uy)）。"""
        raise NotImplementedError

    @property
    def node_dim(self) -> int:
        """节点特征维度（例如 one-hot 类型特征的维度）。"""
        raise NotImplementedError

    @property
    def edge_dim(self) -> int:
        """边特征维度（例如相对位移 (dx, dy)）。"""
        raise NotImplementedError

    @property
    def dt(self) -> float:
        """离散时间步长，用于 ḣ 近似等。"""
        raise NotImplementedError

    @property
    def num_agents(self) -> int:
        """智能体数量（用于 type_states / type_nodes 聚合时的 n_type）。"""
        raise NotImplementedError

    # -------------------------
    # 核心交互接口（子类需要实现）
    # -------------------------
    def reset(self, key: Array) -> GraphsTuple:
        """
        根据随机种子 key 生成新的起始图。
        返回：单图 GraphsTuple（字段需与 gcbfplus.utils.graph.GraphsTuple 对齐）
        """
        raise NotImplementedError

    def step(
        self, graph: GraphsTuple, action: Action
    ) -> Tuple[GraphsTuple, float, float, bool, Dict[str, Any]]:
        """
        执行一步环境转移。
          - 输入：graph（当前图）、action 形状为 (num_agents, action_dim) 的连续动作
          - 输出：next_graph, reward, cost, done, info
            * reward: 标量（可为 Python float 或 0-D jnp.ndarray）
            * cost:   标量（如碰撞率等）
            * done:   是否终止
            * info:   可包含 {"collision": (num_agents,) bool, "reached_goals": (num_agents,) bool, ...}
        """
        raise NotImplementedError

    def forward_graph(self, graph: GraphsTuple, action: Action) -> GraphsTuple:
        """
        纯函数版转移（无副作用、无奖励），供算法在更新时计算 h(x⁺) / ḣ。
        """
        raise NotImplementedError

    def add_edge_feats(self, graph: GraphsTuple, new_state: State) -> GraphsTuple:
        """
        将 graph.states 中的 agent 部分替换为 new_state 对应的 agent 状态，并重构边特征；
        new_state 的拼接顺序应与 GraphsTuple.states 一致：[agents, goals, obstacles]。
        """
        raise NotImplementedError

    def control_affine_dyn(self, agent_state: Array) -> Tuple[Array, Array]:
        """
        返回控制仿射模型 ẋ = f(x) + g(x) u 的 (f, g)：
          - agent_state: (num_agents, state_dim)
          - f: (num_agents, state_dim)
          - g: (num_agents, state_dim, action_dim)
        """
        raise NotImplementedError

    def safe_mask(self, graph: GraphsTuple) -> Array:
        """
        返回形状为 (num_agents,) 的布尔数组：True 表示安全。
        """
        raise NotImplementedError

    def unsafe_mask(self, graph: GraphsTuple) -> Array:
        """
        默认实现：unsafe = ~safe。
        子类如有更高效的实现可以覆写。
        """
        return jnp.logical_not(self.safe_mask(graph))

    def finish_mask(self, graph: GraphsTuple) -> Array:
        """
        返回形状为 (num_agents,) 的布尔数组：True 表示该智能体已到达目标。
        """
        raise NotImplementedError

    def action_lim(self) -> Tuple[Array, Array]:
        """
        连续动作的下界与上界，形如：
          lb: (action_dim,)  ub: (action_dim,)
        算法会按 (num_agents, action_dim) 进行扩展平铺。
        """
        raise NotImplementedError

    def state_lim(self) -> Tuple[Array, Array]:
        """
        状态边界（可用于裁剪或正规化），形如：
          lb: (state_dim,)  ub: (state_dim,)
        """
        raise NotImplementedError

    # -------------------------
    # 参考控制 u_ref（可选钩子）
    # -------------------------
    _u_ref_model: Optional[Callable[[GraphsTuple], Action]] = None

    def set_u_ref_model(self, fn: Optional[Callable[[GraphsTuple], Action]]) -> None:
        """
        注入一个外部参考控制（例如你用 Flax 复刻的 GAT-MF 前向）。
        传入 None 可移除。
        """
        self._u_ref_model = fn

    def u_ref(self, graph: GraphsTuple) -> Action:
        """
        参考控制（形状 (num_agents, action_dim)）。
        默认：若已注入外部模型则调用之；否则由具体环境子类实现（例如朝目标方向）。
        """
        if self._u_ref_model is None:
            raise NotImplementedError(
                "u_ref 未实现且未注入外部模型。请在子类中实现 u_ref，或调用 set_u_ref_model(fn)。"
            )
        return self._u_ref_model(graph)
