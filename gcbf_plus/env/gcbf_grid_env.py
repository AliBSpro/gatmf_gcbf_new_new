# gcbf_grid_env.py
# A single-file, JAX-only grid environment exposing all interfaces used by GCBF / GCBF+.
# No external imports from your repo; includes GraphsTuple + graph processing.

from typing import NamedTuple, Tuple, Optional, List, Union, Dict, Any
import jax
import jax.numpy as jnp
import jax.random as jr
from gcbf_plus.env.base import MultiAgentEnv

# ---------------------------
# Typing aliases (JAX-first)
# ---------------------------
Array = jnp.ndarray
Action = Array
State = Array
Reward = float
Cost = float
Done = bool
Info = Dict[str, Any]
PRNGKey = Array

# ---------------------------
# Graph structure (standalone)
# ---------------------------
class GraphsTuple(NamedTuple):
    # Compatible field names for algo / nets
    n_node: Array      # scalar jnp.array
    n_edge: Array      # scalar jnp.array
    nodes: Array       # (sum_n_node, node_dim)
    edges: Array       # (sum_n_edge, edge_dim)
    states: Array      # (sum_n_node, state_dim)
    receivers: Array   # (sum_n_edge,)
    senders: Array     # (sum_n_edge,)
    node_type: Array   # (sum_n_node,) int32 0=agent,1=goal,2=obs
    env_states: Any    # see EnvState below
    edge_mask: Array   # (sum_n_edge,) bool

    @property
    def is_single(self) -> bool:
        return self.n_node.ndim == 0

    def type_states(self, type_idx: int, n_type: int) -> Array:
        """Gather states by node_type to a dense (n_type, state_dim) array."""
        n_states = self.states.shape[1]
        n_is_type = (self.node_type == type_idx)
        idx = jnp.cumsum(n_is_type) - 1  # positions among selected types
        out = jnp.zeros((n_type, n_states), dtype=self.states.dtype)
        out = out.at[idx, :].add(n_is_type[:, None] * self.states)
        return out

    def type_nodes(self, type_idx: int, n_type: int) -> Array:
        """Gather nodes by node_type to a dense (n_type, node_dim) array."""
        n_feats = self.nodes.shape[1]
        n_is_type = (self.node_type == type_idx)
        idx = jnp.cumsum(n_is_type) - 1  # positions among selected types
        out = jnp.zeros((n_type, n_feats), dtype=self.nodes.dtype)
        out = out.at[idx, :].add(n_is_type[:, None] * self.nodes)
        return out

    def replace(self, **kwargs) -> "GraphsTuple":
        fields = self._fields
        new_vals = []
        for f in fields:
            new_vals.append(kwargs[f] if f in kwargs else getattr(self, f))
        return GraphsTuple(*new_vals)

# ---------------------------
# Environment state container
# ---------------------------
class EnvState(NamedTuple):
    agent: Array  # (n_agent, 2) float32
    goal: Array   # (n_agent, 2) float32
    obs: Array    # (n_obs, 2) float32  (grid cell centers)

# ---------------------------
# Unified grid env (discrete, single-integrator semantics)
# ---------------------------
class UnifiedGridEnv(MultiAgentEnv):
    """
    Discrete grid env (one-cell-per-step) with JAX-friendly, pure-functional graph builders.

    Exposed interfaces (as required by GCBF/GCBF+):
      - reset(key) -> GraphsTuple
      - step(graph, action) -> (GraphsTuple, reward, cost, done, info)
      - forward_graph(graph, action) -> GraphsTuple
      - add_edge_feats(graph, new_state) -> GraphsTuple
      - control_affine_dyn(agent_state) -> (f, g)  # single-integrator: f=0, g=I
      - u_ref(graph) -> (n_agent, 2) reference control toward goals
      - safe_mask(graph) / unsafe_mask(graph) -> (n_agent,) bool
      - finish_mask(graph) -> (n_agent,) bool
      - action_lim() / state_lim()
      - properties: action_dim, node_dim, edge_dim, dt (float)
    """

    # Node type id
    AGENT = 0
    GOAL = 1
    OBS = 2

    # Discrete action indices (used internally for step-like motion)
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
        max_steps: int = 50,
    ):
        self.grid_size = int(grid_size)
        self._num_agents = int(num_agents)  # 使用私有屬性
        self.num_obstacles = int(num_obstacles)
        self.max_steps = int(max_steps)
        self.max_episode_steps = int(max_steps)  # 添加trainer期望的屬性
        self._step_count = 0

        # Dimensions expected by algo/nets (actor/cbf were typically inited with these)
        self._state_dim = 2         # (x, y)
        self._node_dim  = 3         # one-hot: agent/goal/obs
        self._edge_dim  = 2         # relative (dx, dy)
        self._action_dim = 2        # continuous control per agent: (ux, uy)
        self._dt = 1.0              # discrete time step

        # Discrete basis moves (one cell) mapped to continuous vectors
        self._dir_map = jnp.array([
            [ 0.0,  0.0],  # stay
            [ 0.0,  1.0],  # up
            [ 0.0, -1.0],  # down
            [-1.0,  0.0],  # left
            [ 1.0,  0.0],  # right
        ], dtype=jnp.float32)

    # ---------------------------
    # Public properties
    # ---------------------------
    @property
    def num_agents(self) -> int:
        return self._num_agents

    @property
    def dt(self) -> float:
        return self._dt

    @property
    def action_dim(self) -> int:
        return self._action_dim

    @property
    def node_dim(self) -> int:
        return self._node_dim

    @property
    def edge_dim(self) -> int:
        return self._edge_dim

    # ---------------------------
    # Core env API
    # ---------------------------
    def reset(self, key: PRNGKey) -> GraphsTuple:
        """Sample non-overlapping positions for agents, goals, obstacles; return initial graph."""
        self._step_count = 0
        agent_key, goal_key, obs_key = jr.split(key, 3)

        def sample_unique(key, n):
            coords = jnp.stack(jnp.meshgrid(jnp.arange(self.grid_size), jnp.arange(self.grid_size)), axis=-1).reshape(-1, 2)
            # simple random without replacement by shuffling
            idx = jr.permutation(key, coords.shape[0])[:n]
            return coords[idx].astype(jnp.float32)

        # To ensure no overlap across categories, sample larger pool and filter:
        # Simpler: sample sequentially while masking previous picks.
        pool = jnp.stack(jnp.meshgrid(jnp.arange(self.grid_size), jnp.arange(self.grid_size)), axis=-1).reshape(-1, 2).astype(jnp.float32)

        # agents
        idx = jr.permutation(agent_key, pool.shape[0])
        agent = pool[idx[:self.num_agents]]
        rest  = pool[idx[self.num_agents:]]

        # goals
        idx2 = jr.permutation(goal_key, rest.shape[0])
        goal = rest[idx2[:self.num_agents]]
        rest = rest[idx2[self.num_agents:]]

        # obs
        idx3 = jr.permutation(obs_key, rest.shape[0])
        obs = rest[idx3[:self.num_obstacles]]

        env_state = EnvState(agent=agent, goal=goal, obs=obs)
        return self._build_graph(env_state)

    def step(self, graph: GraphsTuple, action: Array) -> Tuple[GraphsTuple, Reward, Cost, Done, Info]:
        """
        One synchronous step. For GCBF/GCBF+, 'action' is continuous (n_agent, 2); we:
          1) map to discrete 5-way moves (stay/U/D/L/R) by dominant axis;
          2) propose new cells, clip to boundary;
          3) resolve collisions (agent-agent duplicates & hitting obstacles) → revert to old cell;
          4) rebuild graph; compute dense reward & scalar cost.
        Reward = -MSE(u_applied, u_ref); Cost = mean(collision_flags after move).
        """
        self._step_count += 1

        agent   = graph.type_states(self.AGENT, self.num_agents)
        goal    = graph.type_states(self.GOAL,  self.num_agents)
        obs     = graph.env_states.obs

        # 1) map continuous action -> discrete indices
        disc = _cont2disc(action)  # (n_agent,) int32 ∈ {0..4}

        # 2~3) apply moves with collision resolution
        new_agent = _apply_discrete_moves(agent, disc, obs, self.grid_size)

        # 4) build next graph
        next_graph = self._build_graph(EnvState(agent=new_agent, goal=goal, obs=obs))

        # reward & cost
        u_applied = action.astype(jnp.float32)
        uref = self.u_ref(graph)  # (n_agent,2)
        reward = -jnp.mean(jnp.sum((u_applied - uref) ** 2, axis=-1))
        flags = _collision_flags(new_agent, obs)  # (n_agent,)
        cost = jnp.mean(flags.astype(jnp.float32))

        done = (self._step_count >= self.max_steps) or jnp.all(self.finish_mask(next_graph))
        info = {
            "step_count": self._step_count,
            "collision": flags,                               # (n_agent,) bool
            "reached_goals": self.finish_mask(next_graph),    # (n_agent,) bool
        }
        return next_graph, reward, cost, done, info

    # ---------------------------
    # Interfaces called by GCBF / GCBFPlus
    # ---------------------------
    def forward_graph(self, graph: GraphsTuple, action: Array) -> GraphsTuple:
        """
        Side-effect-free forward: same transition semantics as step (no reward/cost/done).
        Used for h(x+), ḣ ≈ (h(x+)-h(x))/dt in algo updates.
        """
        agent = graph.type_states(self.AGENT, self.num_agents)
        goal  = graph.type_states(self.GOAL,  self.num_agents)
        obs   = graph.env_states.obs
        disc = _cont2disc(action)
        new_agent = _apply_discrete_moves(agent, disc, obs, self.grid_size)
        return self._build_graph(EnvState(agent=new_agent, goal=goal, obs=obs))

    def add_edge_feats(self, graph: GraphsTuple, new_state: Array) -> GraphsTuple:
        """
        Replace agent states in `graph.states` with `new_state`'s agent part, then rebuild edges.
        Expected `new_state` layout = [agents(na), goals(na), obs(no)] along dim 0 (same as graph.states).
        """
        na = self.num_agents
        no = graph.env_states.obs.shape[0]
        new_agent = new_state[:na, :]
        new_goal  = new_state[na:na+na, :]
        new_obs   = new_state[na+na:na+na+no, :]
        return self._build_graph(EnvState(agent=new_agent, goal=new_goal, obs=new_obs))

    def control_affine_dyn(self, agent_state: Array) -> Tuple[Array, Array]:
        """
        Single-integrator continuous surrogate: x_dot = u  =>  f(x)=0, g(x)=I.
        agent_state: (n_agent, 2)
        returns:
          f: (n_agent, 2)
          g: (n_agent, 2, 2)
        """
        n = agent_state.shape[0]
        f = jnp.zeros_like(agent_state)
        g = jnp.tile(jnp.eye(2, dtype=jnp.float32)[None, :, :], (n, 1, 1))
        return f, g

    def u_ref(self, graph: GraphsTuple) -> Array:
        """
        Reference control: clipped direction to goals in {-1..1} on each axis.
        """

        if getattr(self, "_u_ref_model", None) is not None:
            return self._u_ref_model(graph)  # (n_agent, 2)
        agent = graph.type_states(self.AGENT, self.num_agents)
        goal  = graph.type_states(self.GOAL,  self.num_agents)
        diff = goal - agent
        # clamp each axis to [-1, 1] to match 'one-cell-per-step' continuous surrogate
        return jnp.clip(diff, -1.0, 1.0)

    def safe_mask(self, graph: GraphsTuple) -> Array:
        """Per-agent boolean: True if NOT colliding (see _collision_flags)."""
        agent = graph.type_states(self.AGENT, self.num_agents)
        obs   = graph.env_states.obs
        return ~_collision_flags(agent, obs)

    def unsafe_mask(self, graph: GraphsTuple) -> Array:
        return ~self.safe_mask(graph)

    def finish_mask(self, graph: GraphsTuple) -> Array:
        """Per-agent boolean: True if agent is exactly on its goal."""
        agent = graph.type_states(self.AGENT, self.num_agents)
        goal  = graph.type_states(self.GOAL,  self.num_agents)
        return jnp.all(jnp.isclose(agent, goal), axis=-1)

    def action_lim(self) -> Tuple[Array, Array]:
        """Per-agent continuous control limits; algo will tile to all agents."""
        lb = jnp.array([-1.0, -1.0], dtype=jnp.float32)
        ub = jnp.array([ 1.0,  1.0], dtype=jnp.float32)
        return lb, ub

    def state_lim(self) -> Tuple[Array, Array]:
        lb = jnp.array([0.0, 0.0], dtype=jnp.float32)
        ub = jnp.array([self.grid_size - 1, self.grid_size - 1], dtype=jnp.float32)
        return lb, ub

    # ---------------------------
    # Graph builders
    # ---------------------------
    def _build_graph(self, env_state: EnvState) -> GraphsTuple:
        """
        Layout:
          - nodes: one-hot type feats: [agent]*na + [goal]*na + [obs]*no   (dim=3)
          - states: concatenated positions in same order (dim=2)
          - edges: directed; concatenate A->A (L1<=2, no self), A->G (paired), A->O (L1<=2)
        """
        na = env_state.agent.shape[0]
        no = env_state.obs.shape[0]
        nnodes = 2 * na + no

        # node features one-hot
        nodes = jnp.zeros((nnodes, self._node_dim), dtype=jnp.float32)
        nodes = nodes.at[:na, 0].set(1.0)          # agents
        nodes = nodes.at[na:2*na, 1].set(1.0)      # goals
        nodes = nodes.at[2*na:, 2].set(1.0)        # obs

        # node types
        node_type = jnp.zeros((nnodes,), dtype=jnp.int32)
        node_type = node_type.at[na:2*na].set(self.GOAL)
        node_type = node_type.at[2*na:].set(self.OBS)

        # states (x, y)
        states = jnp.concatenate([env_state.agent, env_state.goal, env_state.obs], axis=0).astype(jnp.float32)

        # build edges & features
        senders, receivers, e_feats, edge_mask = self._build_edges(env_state)

        return GraphsTuple(
            n_node=jnp.array(nnodes),
            n_edge=jnp.array(senders.shape[0]),
            nodes=nodes,
            edges=e_feats.astype(jnp.float32),
            states=states,
            receivers=receivers.astype(jnp.int32),
            senders=senders.astype(jnp.int32),
            node_type=node_type,
            env_states=env_state,
            edge_mask=edge_mask,  # 新增
        )

    def _build_edges(self, env_state: EnvState) -> Tuple[Array, Array, Array, Array]:
        """
        Build edges as concatenation of three blocks:
          - A->A: L1 <= 2, no self
          - A->G: each agent to its paired goal
          - A->O: L1 <= 2
        Edge feature = receiver_pos - sender_pos = (dx, dy).
        """
        na = env_state.agent.shape[0]
        no = env_state.obs.shape[0]

        # indices
        ids_A = jnp.arange(0, na, dtype=jnp.int32)
        ids_G = jnp.arange(na, 2*na, dtype=jnp.int32)
        ids_O = jnp.arange(2*na, 2*na + no, dtype=jnp.int32)

        # positions by slice from concatenated states
        A = env_state.agent
        G = env_state.goal
        O = env_state.obs

        blocks = []

        # A->A
        # For every pair (i (recv), j (send)), connect if manhattan <= 2 and i != j.
        diff_AA = A[:, None, :] - A[None, :, :]                   # (na, na, 2)
        man_AA  = jnp.abs(diff_AA[..., 0]) + jnp.abs(diff_AA[..., 1])
        mask_AA = (man_AA > 0.0) & (man_AA <= 2.0)                # (na, na)
        s_AA, r_AA, f_AA, m_AA = _mask_to_edges(ids_A, ids_A, diff_AA, mask_AA)

        # A->G (paired)
        diff_AG = A[:, None, :] - G[None, :, :]                   # (na, na, 2)
        mask_AG = jnp.eye(na, dtype=jnp.bool_)                    # only pair (i->i)
        s_AG, r_AG, f_AG, m_AG = _mask_to_edges(ids_A, ids_G, diff_AG, mask_AG)

        # A->O within L1<=2
        if no > 0:
            diff_AO = A[:, None, :] - O[None, :, :]               # (na, no, 2)
            man_AO  = jnp.abs(diff_AO[..., 0]) + jnp.abs(diff_AO[..., 1])
            mask_AO = (man_AO <= 2.0)
            s_AO, r_AO, f_AO, m_AO = _mask_to_edges(ids_A, ids_O, diff_AO, mask_AO)
            senders = jnp.concatenate([s_AA, s_AG, s_AO], axis=0)
            recvers = jnp.concatenate([r_AA, r_AG, r_AO], axis=0)
            feats   = jnp.concatenate([f_AA, f_AG, f_AO], axis=0)
            mask_1d = jnp.concatenate([m_AA, m_AG, m_AO], axis=0)
        else:
            senders = jnp.concatenate([s_AA, s_AG], axis=0)
            recvers = jnp.concatenate([r_AA, r_AG], axis=0)
            feats   = jnp.concatenate([f_AA, f_AG], axis=0)
            mask_1d = jnp.concatenate([m_AA, m_AG], axis=0)

        return senders, recvers, feats, mask_1d

    def set_u_ref_model(self, fn):
        """fn: Callable[[GraphsTuple], jnp.ndarray of shape (n_agent, 2)]"""
        self._u_ref_model = fn


# ---------------------------
# Helper functions (JAX-pure)
# ---------------------------
def _cont2disc(u: Array) -> Array:
    """
    Map continuous action (n_agent,2) -> discrete indices {0..4} by dominant axis rule.
    stay if ||u|| < 1e-6; else choose axis with Union[larger, component]| and sign.
    """
    u = u.astype(jnp.float32)
    dx, dy = u[:, 0], u[:, 1]
    mag = jnp.sqrt(dx * dx + dy * dy)
    stay = mag < 1e-6
    horiz = (jnp.abs(dx) >= jnp.abs(dy))
    disc = jnp.full((u.shape[0],), UnifiedGridEnv.ACTION_STAY, dtype=jnp.int32)
    disc = jnp.where((~stay) & horiz & (dx > 0), UnifiedGridEnv.ACTION_RIGHT, disc)
    disc = jnp.where((~stay) & horiz & (dx < 0), UnifiedGridEnv.ACTION_LEFT,  disc)
    disc = jnp.where((~stay) & (~horiz) & (dy > 0), UnifiedGridEnv.ACTION_UP,   disc)
    disc = jnp.where((~stay) & (~horiz) & (dy < 0), UnifiedGridEnv.ACTION_DOWN, disc)
    return disc

def _apply_discrete_moves(agent: Array, disc: Array, obs: Array, grid_size: int) -> Array:
    """
    Resolve proposed moves with obstacle & agent-duplicate checks.
    If a proposed cell is an obstacle or duplicates a previously accepted agent cell,
    the agent stays at its current cell.
    """
    dir_map = jnp.array([
        [ 0.0,  0.0],
        [ 0.0,  1.0],
        [ 0.0,- 1.0],
        [-1.0,  0.0],
        [ 1.0,  0.0],
    ], dtype=jnp.float32)

    n = agent.shape[0]
    prop = jnp.clip(agent + dir_map[disc], 0.0, grid_size - 1)  # (n,2)
    new_states = agent  # will be updated sequentially
    committed = jnp.zeros((n,), dtype=jnp.bool_)

    def body(i, carry):
        ns, cm = carry
        cand = prop[i]

        # obstacle hit? (exact grid-cell equality)
        hit_obs = (obs.shape[0] > 0) & jnp.any(jnp.all(jnp.isclose(obs, cand[None, :]), axis=-1))

        # duplicates with already committed agents?
        same_as = jnp.all(jnp.isclose(ns, cand[None, :]), axis=-1) & cm
        hit_dup = jnp.any(same_as)

        collide = jnp.logical_or(hit_obs, hit_dup)
        new_i = jnp.where(collide, agent[i], cand)
        ns = ns.at[i].set(new_i)
        cm = cm.at[i].set(True)
        return (ns, cm)

    new_states, _ = jax.lax.fori_loop(0, n, body, (new_states, committed))
    return new_states

def _collision_flags(agent: Array, obs: Array) -> Array:
    """
    Per-agent collision = near other agents (manhattan <= 1) OR on an obstacle cell.
    """
    n = agent.shape[0]
    # A-A (L1 <= 1, excluding self)
    diff = agent[:, None, :] - agent[None, :, :]            # (n,n,2)
    man  = jnp.abs(diff[..., 0]) + jnp.abs(diff[..., 1])    # (n,n)
    aa   = (man <= 1.0) & (~jnp.eye(n, dtype=jnp.bool_))

    if obs.shape[0] > 0:
        diff_ao = agent[:, None, :] - obs[None, :, :]  # (n, no, 2)
        man_ao = jnp.abs(diff_ao[..., 0]) + jnp.abs(diff_ao[..., 1])  # (n, no)
        ao = jnp.any(man_ao <= 1.0, axis=1)  # (n,)
    else:
        ao = jnp.zeros((n,), dtype=jnp.bool_)

    return jnp.logical_or(jnp.any(aa, axis=1), ao)

def _mask_to_edges(ids_recv: Array, ids_send: Array, diffs: Array, mask: Array) -> Tuple[Array, Array, Array, Array]:
    """
    Convert (recv, send) mask to edge lists and relative feature (dx,dy)=recv - send.
    diffs is assumed recv_pos - send_pos with shape (n_recv, n_send, 2).
    """
    # 使用固定大小的方法避免ConcretizationTypeError
    n_recv, n_send = mask.shape
    max_edges = n_recv * n_send
    
    # 创建所有可能的邊索引
    all_i = jnp.repeat(jnp.arange(n_recv), n_send)
    all_j = jnp.tile(jnp.arange(n_send), n_recv)
    
    # 展平mask並选择有效邊
    mask_flat = mask.flatten()
    valid_edges = jnp.where(mask_flat, size=max_edges, fill_value=0)[0]
    
    # 使用有效邊数量（但保持固定形状）
    n_true = jnp.sum(mask_flat).astype(jnp.int32)
    
    # 选择有效的索引
    valid_i = all_i[valid_edges]
    valid_j = all_j[valid_edges]
    
    # 创建固定大小的输出
    senders = ids_send[valid_j]
    receivers = ids_recv[valid_i]
    feats = diffs[valid_i, valid_j]
    edge_mask = jnp.arange(max_edges) < n_true
    
    return senders, receivers, feats, edge_mask
