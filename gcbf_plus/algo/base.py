# gcbf_plus/algo/base.py
class MultiAgentController:
    def __init__(self, env, node_dim, edge_dim, action_dim, n_agents):
        self._env = env
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.action_dim = action_dim
        self.n_agents = n_agents
