

# grid_model.py
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import NamedTuple, Tuple, Optional, List, Dict, Any, Union


# ============================================================
# Torch graph structures (port of the uploaded graph API)
# ============================================================

class EdgeBlock(NamedTuple):
    edge_feats: torch.Tensor   # (n_recv, n_send, 2)
    edge_mask: torch.Tensor    # (n_recv, n_send) bool
    ids_recv: torch.Tensor     # (n_recv,) long
    ids_send: torch.Tensor     # (n_send,) long

    @property
    def n_recv(self):
        return int(self.ids_recv.numel())

    @property
    def n_send(self):
        return int(self.ids_send.numel())

    @property
    def n_edges(self):
        return self.n_recv * self.n_send


class GraphsTuple(NamedTuple):
    n_node: torch.Tensor     # scalar long
    n_edge: torch.Tensor     # scalar long
    nodes: torch.Tensor      # (sum_n_node, node_dim=3)
    edges: torch.Tensor      # (sum_n_edge, edge_dim=2)
    states: torch.Tensor     # (sum_n_node, 2)
    receivers: torch.Tensor  # (sum_n_edge,) long
    senders: torch.Tensor    # (sum_n_edge,) long
    node_type: torch.Tensor  # (sum_n_node,) long
    env_states: Any          # TorchEnvState

    @property
    def is_single(self) -> bool:
        return True

    def type_states(self, type_idx: int, n_type: int) -> torch.Tensor:
        # Returns (n_type, state_dim) in the same ordering as creation
        mask = (self.node_type == type_idx)
        feats = self.states[mask]
        assert feats.shape[0] == n_type, f"Expected {n_type} nodes of type {type_idx}, got {feats.shape[0]}"
        return feats


@dataclass
class GridObstacle:
    positions: torch.Tensor  # (n_obs, 2) long


@dataclass
class EnvState:
    agent: torch.Tensor   # (n_agent, 2) long
    goal: torch.Tensor    # (n_agent, 2) long
    obstacle: GridObstacle


# ============================================================
# Pure-PyTorch Grid Environment with Graph builder
# ============================================================

class UnifiedGridEnvTorch:
    AGENT = 0
    GOAL = 1
    OBS = 2

    # ----- five discrete actions (with STAY) -----
    ACTION_STAY  = 0
    ACTION_UP    = 1
    ACTION_DOWN  = 2
    ACTION_LEFT  = 3
    ACTION_RIGHT = 4

    def __init__(
        self,
        grid_size: int = 3,
        num_agents: int = 2,
        num_obstacles: int = 1,
        device: Optional[torch.device] = None,
        max_steps: int = 30,
        seed: Optional[int] = None,
    ):
        self.grid_size = int(grid_size)
        self.num_agents = int(num_agents)
        self.num_obstacles = int(num_obstacles)
        self.max_steps = int(max_steps)
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.rng = np.random.RandomState(seed if seed is not None else 0)

        self._step = 0
        self._episode_return = 0.0

        self.graph: Optional[GraphsTuple] = None
        self._potential: Optional[torch.Tensor] = None  # (N,1)

    # ---------------- core helpers ----------------

    def _sample_unique_positions(self, k: int) -> torch.Tensor:
        chosen = set()
        coords = []
        while len(coords) < k:
            x = int(self.rng.randint(0, self.grid_size))
            y = int(self.rng.randint(0, self.grid_size))
            if (x, y) not in chosen:
                chosen.add((x, y))
                coords.append((x, y))
        return torch.tensor(coords, dtype=torch.long, device=self.device)

    def _compute_potential(self, agent_xy: torch.Tensor, goal_xy: torch.Tensor) -> torch.Tensor:
        # Φ = -L1
        l1 = (agent_xy - goal_xy).abs().sum(dim=-1, keepdim=True).to(torch.float32)
        return -l1

    # ---------------- public API ----------------

    @torch.no_grad()
    def reset(self, seed: Optional[int] = None) -> GraphsTuple:
        if seed is not None:
            self.rng = np.random.RandomState(seed)
        self._step = 0
        self._episode_return = 0.0

        coords = self._sample_unique_positions(self.num_agents*2 + self.num_obstacles)
        agent = coords[: self.num_agents]
        goal = coords[self.num_agents : 2*self.num_agents]
        obs  = coords[2*self.num_agents :]

        env_state = EnvState(agent=agent, goal=goal, obstacle=GridObstacle(positions=obs))
        self.graph = self._build_graph(env_state)
        self._potential = self._compute_potential(agent, goal)
        return self.graph

    @torch.no_grad()
    def step(self, action: Union[torch.Tensor, np.ndarray, List[int]]) -> Tuple[GraphsTuple, torch.Tensor]:
        """
        One step with discrete actions or with logits/probs (N,5).
        Accepts:
          - (N,5) logits/probs → argmax to {STAY,UP,DOWN,LEFT,RIGHT}
          - (N,)  integer indices in {0..4}
        Returns: (new_graph, per-agent reward (N,1))
        """
        assert self.graph is not None, "Call reset first"
        N = self.num_agents

        if not isinstance(action, torch.Tensor):
            action = torch.as_tensor(action, device=self.device)

        if action.dim() == 2 and action.shape[-1] == 5:
            # 🎯 修复：使用概率採樣而非argmax，避免初期站樁
            probs = F.softmax(action, dim=-1)
            a_idx = torch.multinomial(probs, 1).squeeze(-1)  # (N,) 概率採樣
        elif action.dim() == 1:
            a_idx = action.to(torch.long)
        else:
            raise ValueError("action must be (N,5) or (N,)")

        # move
        move = torch.zeros((N, 2), dtype=torch.long, device=self.device)
        # STAY: (0,0) — already default
        move[a_idx == self.ACTION_UP, 1] = 1
        move[a_idx == self.ACTION_DOWN, 1] = -1
        move[a_idx == self.ACTION_LEFT, 0] = -1
        move[a_idx == self.ACTION_RIGHT, 0] = 1

        agent = self.graph.env_states.agent + move
        agent = agent.clamp(0, self.grid_size - 1)

        # obstacles block
        obs = self.graph.env_states.obstacle.positions
        if obs.numel() > 0:
            for i in range(N):
                if (obs == agent[i]).all(dim=1).any():
                    agent[i] = self.graph.env_states.agent[i]

        goal = self.graph.env_states.goal

        new_potential = self._compute_potential(agent, goal)
        reward = new_potential - self._potential  # (N,1)

        # rebuild graph
        env_state = EnvState(agent=agent, goal=goal, obstacle=GridObstacle(positions=obs))
        self.graph = self._build_graph(env_state)

        self._potential = new_potential
        self._step += 1
        self._episode_return += float(reward.mean().item())
        return self.graph, reward

    # ---------------- features / adjacency for GAT-MF ----------------

    @torch.no_grad()
    def state_features(self, graph: Optional[GraphsTuple] = None) -> torch.Tensor:
        """(N,5): [dx, dy, |dx|, |dy|, 1]"""
        g = self.graph if graph is None else graph
        agent = g.type_states(self.AGENT, self.num_agents).to(torch.float32)
        goal = g.type_states(self.GOAL, self.num_agents).to(torch.float32)
        dxdy = goal - agent
        dx, dy = dxdy[:, 0:1], dxdy[:, 1:2]
        ones = torch.ones_like(dx)
        return torch.cat([dx, dy, dx.abs(), dy.abs(), ones], dim=1)

    @torch.no_grad()
    def adjacency_local(self, graph: Optional[GraphsTuple] = None) -> torch.Tensor:
        """A-A 4-neighbor adjacency (N,N), zeros diag."""
        g = self.graph if graph is None else graph
        agent = g.type_states(self.AGENT, self.num_agents)
        diff = (agent[:, None, :] - agent[None, :, :]).abs().sum(dim=-1)  # (N,N) L1
        G = (diff == 1).to(torch.float32)
        G.fill_diagonal_(0.0)
        return G

    @torch.no_grad()
    def adjacency_fully_connected(self) -> torch.Tensor:
        """Fully-connected without self loops (N,N)."""
        N = self.num_agents
        G = torch.ones((N, N), dtype=torch.float32, device=self.device)
        G.fill_diagonal_(0.0)
        return G

    @property
    def potential(self) -> torch.Tensor:
        """(N,1) Φ = -L1"""
        return self._potential.clone()

    @property
    def episode_return(self) -> float:
        return self._episode_return

    # ---------------- graph builder (follow uploaded structure, but Torch) ----------------

    def _build_graph(self, env_state: EnvState) -> GraphsTuple:
        n_nodes = 2*self.num_agents + self.num_obstacles

        # node features one-hot: [agent, goal, obs]
        nodes = torch.zeros((n_nodes, 3), dtype=torch.float32, device=self.device)
        nodes[:self.num_agents, 0] = 1.0
        nodes[self.num_agents:2*self.num_agents, 1] = 1.0
        if self.num_obstacles > 0:
            nodes[2*self.num_agents:, 2] = 1.0

        node_type = torch.zeros((n_nodes,), dtype=torch.long, device=self.device)
        node_type[self.num_agents:2*self.num_agents] = self.GOAL
        if self.num_obstacles > 0:
            node_type[2*self.num_agents:] = self.OBS

        states = torch.cat([
            env_state.agent.to(torch.float32),
            env_state.goal.to(torch.float32),
            env_state.obstacle.positions.to(torch.float32) if self.num_obstacles > 0 else torch.empty((0,2), device=self.device),
        ], dim=0)

        # Edge blocks (A->A with L1<=2; A->G pair; A->O L1<=2)
        blocks: List[EdgeBlock] = []

        id_agent = torch.arange(self.num_agents, device=self.device, dtype=torch.long)
        id_goal  = torch.arange(self.num_agents, device=self.device, dtype=torch.long) + self.num_agents

        agent_pos = env_state.agent.to(torch.float32)
        goal_pos = env_state.goal.to(torch.float32)
        # A->A
        diff_aa = agent_pos[:, None, :] - agent_pos[None, :, :]  # (recv, send, 2)
        manh = diff_aa.abs().sum(dim=-1)
        mask_aa = (manh > 0) & (manh <= 2)
        blocks.append(EdgeBlock(edge_feats=diff_aa, edge_mask=mask_aa, ids_recv=id_agent, ids_send=id_agent))

        # A->G one-to-one
        diff_ag = agent_pos[:, None, :] - goal_pos[None, :, :]
        mask_ag = torch.eye(self.num_agents, dtype=torch.bool, device=self.device)
        blocks.append(EdgeBlock(edge_feats=diff_ag, edge_mask=mask_ag, ids_recv=id_agent, ids_send=id_goal))

        # A->O L1<=2
        if self.num_obstacles > 0:
            id_obs = torch.arange(self.num_obstacles, device=self.device, dtype=torch.long) + 2*self.num_agents
            obs_pos = env_state.obstacle.positions.to(torch.float32)
            diff_ao = agent_pos[:, None, :] - obs_pos[None, :, :]
            manh_ao = diff_ao.abs().sum(dim=-1)
            mask_ao = (manh_ao <= 2)
            blocks.append(EdgeBlock(edge_feats=diff_ao, edge_mask=mask_ao, ids_recv=id_agent, ids_send=id_obs))

        senders, receivers, edge_feats = self._edge_blocks_to_arrays(blocks)

        return GraphsTuple(
            n_node=torch.tensor(n_nodes, dtype=torch.long, device=self.device),
            n_edge=torch.tensor(senders.numel(), dtype=torch.long, device=self.device),
            nodes=nodes,
            edges=edge_feats,
            states=states,
            receivers=receivers,
            senders=senders,
            node_type=node_type,
            env_states=env_state
        )

    def _edge_blocks_to_arrays(self, blocks: List[EdgeBlock]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not blocks:
            zlong = torch.zeros((0,), dtype=torch.long, device=self.device)
            zfeats = torch.zeros((0,2), dtype=torch.float32, device=self.device)
            return zlong, zlong, zfeats

        senders_list, receivers_list, feats_list = [], [], []
        for b in blocks:
            ii, jj = torch.where(b.edge_mask)
            if ii.numel() == 0:
                continue
            senders_list.append(b.ids_send[jj])
            receivers_list.append(b.ids_recv[ii])
            feats_list.append(b.edge_feats[ii, jj])

        if not feats_list:
            zlong = torch.zeros((0,), dtype=torch.long, device=self.device)
            zfeats = torch.zeros((0,2), dtype=torch.float32, device=self.device)
            return zlong, zlong, zfeats

        senders = torch.cat(senders_list, dim=0)
        receivers = torch.cat(receivers_list, dim=0)
        feats = torch.cat(feats_list, dim=0).to(torch.float32)
        return senders, receivers, feats


# ============================================================
# Conversion helpers for GAT-MF
# ============================================================

@torch.no_grad()
def gcbf_to_gatmf_format(graph: GraphsTuple, env: UnifiedGridEnvTorch) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (state_tensor: (1,N,5), adj: (1,N,N))"""
    s = env.state_features(graph)  # (N,5)
    G = env.adjacency_local(graph)  # (N,N) or choose fully-connected if desired
    return s.unsqueeze(0), G.unsqueeze(0)


# ============================================================
# Adapter with GAT-MF training interface
# ============================================================

class Model:
    """
    Exposes the exact interface used by grid-train for GAT-MF:

      - init_exogenous_variables(num_grid, num_diamond, diamond_extent, num_move)
      - init_endogenous_variables()
      - Gmat  (numpy float32 NxN)
      - output_record() -> numpy (N,5)
      - get_reward() -> numpy (N,1)  (potential Φ = -L1)
      - move_miner(action_vector: (N,5) logits/probs or (N,) ints in {0..4})
      - get_return() -> float
    """
    def __init__(self, *, grid_size_default: int = 3, num_agents_default: Optional[int] = 2,
                 num_obstacles_default: int = 1, device: Optional[str] = None, fully_connected_adj: bool = False,
                 max_steps_default: int = 30, seed_default: Optional[int] = None):
        self.device = torch.device(device) if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.grid_size_default = int(grid_size_default)
        self.num_agents_default = int(num_agents_default) if num_agents_default is not None else None
        self.num_obstacles_default = int(num_obstacles_default)
        self.fully_connected_adj = bool(fully_connected_adj)
        self.max_steps_default = int(max_steps_default)
        self.seed_default = seed_default

        self.env: Optional[UnifiedGridEnvTorch] = None
        self.graph: Optional[GraphsTuple] = None
        self.Gmat = None  # numpy (N,N)
        self._episode_return = 0.0
        
        # 自动初始化环境
        self.init_exogenous_variables(self.grid_size_default)

    def init_exogenous_variables(self, num_grid: int, num_diamond=None, diamond_extent=None, num_move=None):
        Ngrid = int(num_grid)
        Nagents = self.num_agents_default if self.num_agents_default is not None else Ngrid
        self.env = UnifiedGridEnvTorch(
            grid_size=Ngrid,
            num_agents=Nagents,
            num_obstacles=self.num_obstacles_default,
            device=self.device,
            max_steps=self.max_steps_default,
            seed=self.seed_default
        )
        self.graph = self.env.reset(seed=int(np.random.randint(0, 2**31-1)))

        # Build once Gmat (constant) to match original trainer expectations
        # 注意：这里只是初始化，真正的动态邻接会在get_dynamic_adjacency中计算
        if self.fully_connected_adj:
            G = self.env.adjacency_fully_connected()
        else:
            G = self.env.adjacency_local(self.graph)
        self.Gmat = G.detach().cpu().numpy().astype(np.float32)
        
        # 添加动态邻接支持
        self.use_dynamic_adjacency = True  # 启用动态邻接
        print("🔗 启用动态局部连接")
    
    def get_dynamic_adjacency(self, as_tensor=False):
        """获取基于当前位置的动态邻接矩阵"""
        try:
            if self.fully_connected_adj:
                # 全连接始终相同
                G = self.env.adjacency_fully_connected()
            else:
                # 基于当前图状态的局部邻接
                G = self.env.adjacency_local(self.graph)
            
            if as_tensor:
                return G
            else:
                return G.detach().cpu().numpy().astype(np.float32)
                
        except Exception as e:
            print(f"⚠️  动态邻接计算失败，使用静态: {e}")
            # 回退到静态邻接
            if as_tensor:
                return torch.from_numpy(self.Gmat).to(self.env.device)
            else:
                return self.Gmat

    def init_endogenous_variables(self):
        self.graph = self.env.reset(seed=int(np.random.randint(0, 2**31-1)))

    @property
    def output_dim(self):
        """返回状态特征的维度，从output_record()可知是5维"""
        return 5

    def reset(self, seed: Optional[int] = None):
        """重置环境"""
        if self.env is not None:
            if seed is None:
                seed = int(np.random.randint(0, 2**31-1))
            self.graph = self.env.reset(seed=seed)
            self._episode_return = 0.0
        return self.graph

    def output_record(self):
        s = self.env.state_features(self.graph)  # (N,5)
        return s.detach().cpu().numpy().astype(np.float32)

    def get_reward(self):
        # Return the potential Φ(s) (trainer uses diff to get per-step reward)
        phi = self.env.potential  # (N,1)
        return phi.detach().cpu().numpy().astype(np.float32)

    def move_miner(self, action_vector: Union[np.ndarray, torch.Tensor, List[int]]):
        """
        Accept (N,5) logits/probs or (N,) indices {0..4}:
          0: STAY, 1: UP, 2: DOWN, 3: LEFT, 4: RIGHT
        """
        self.graph, step_r = self.env.step(action_vector)  # (N,1)
        self._episode_return += float(step_r.mean().item())

    def get_return(self):
        return float(self._episode_return)
