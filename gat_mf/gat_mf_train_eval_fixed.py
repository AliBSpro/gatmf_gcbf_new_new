import os
import sys
# 确保能找到当前目录的模块
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# external packages
import numpy as np
import time
import torch
import copy
import json
from tqdm import tqdm
import torch.nn.functional as F

# self writing files
import grid_networks as networks
import grid_model


class Buffer:
    def __init__(self, size, N, S_dim, A_dim, device):
        self.size = size
        self.N = N  # 智能体数量
        self.S_dim = S_dim
        self.A_dim = A_dim
        self.device = device

        self.s = np.zeros((size, S_dim), dtype=np.float32)    # state
        self.s1 = np.zeros((size, S_dim), dtype=np.float32)   # next state
        self.a = np.zeros((size, A_dim), dtype=np.float32)    # action (logits or one-hot)
        self.r = np.zeros((size, N), dtype=np.float32)        # 🔧 修复：个体化奖励 (size, N)
        self.end = np.ones((size, N), dtype=np.float32)       # 🔧 修复：个体化end_mask (size, N)

        self.pointer = 0
        self.now_size = 0

        # move to torch tensors on device
        self.s = torch.from_numpy(self.s).to(self.device)
        self.s1 = torch.from_numpy(self.s1).to(self.device)
        self.a = torch.from_numpy(self.a).to(self.device)
        self.r = torch.from_numpy(self.r).to(self.device)
        self.end = torch.from_numpy(self.end).to(self.device)

    def add(self, s, a, r, s1, end):
        self.s[self.pointer] = s
        if isinstance(a, np.ndarray):
            a = torch.from_numpy(a).to(self.device)
        self.a[self.pointer] = a
        # r和end现在都是(N,)形状，支持个体化
        self.r[self.pointer] = r
        self.s1[self.pointer] = s1
        if isinstance(end, np.ndarray):
            end = torch.from_numpy(end).to(self.device)
        self.end[self.pointer] = end

        self.pointer = (self.pointer + 1) % self.size
        self.now_size = min(self.now_size + 1, self.size)

    def sample(self, batch_size):
        indices = np.random.choice(self.now_size, size=batch_size, replace=False)
        s_batch = self.s[indices]
        a_batch = self.a[indices]
        r_batch = self.r[indices]
        s1_batch = self.s1[indices]
        end_batch = self.end[indices]
        return s_batch, a_batch, r_batch, s1_batch, end_batch


class MARL:    # Multi-agent reinforcement learning
    def __init__(self,
                 # —— 环境结构 ——
                 num_grid=3,
                 num_agents=2,
                 num_obstacles=1,
                 env_max_steps=50,
                 env_seed=123,
                 fully_connected_adj=False,  # 改为局部连接

                 # —— 采样/训练设置 ——
                 max_steps=80,
                 max_episode=1500,
                 update_batch=32,
                 batch_size=128,
                 buffer_capacity=150_000,
                 update_interval=4,
                 save_interval=100,
                 eval_interval=25,
                 eval_episodes=50,

                 # —— 优化/稳定性 ——
                 lr=5e-5,
                 lr_decay=True,
                 grad_clip=True,
                 max_grad_norm=2.0,
                 soft_replace_rate=0.01,
                 gamma=0.85,

                 # —— 探索策略 ——
                 explore_noise=0.1,
                 explore_noise_decay=True,
                 explore_decay=0.998,
                 explore_noise_min=0.01,

                 # —— 奖励/到达阈值 ——
                 arrival_bonus=15.0,
                 arrival_tol=1.0,
                 ):
        """
        GAT-MF 多智能体强化学习训练器
        """
        # 保存配置
        self.num_grid = num_grid
        self.N = num_agents
        self.num_obstacles = num_obstacles
        self.env_max_steps = env_max_steps
        self.env_seed = env_seed
        self.fully_connected_adj = fully_connected_adj
        
        self.max_steps = max_steps
        self.max_episode = max_episode
        self.update_batch = update_batch
        self.batch_size = batch_size
        self.buffer_capacity = buffer_capacity
        self.update_interval = update_interval
        self.save_interval = save_interval
        self.eval_interval = eval_interval
        self.eval_episodes = eval_episodes

        self.lr = lr
        self.lr_decay = lr_decay
        self.grad_clip = grad_clip
        self.max_grad_norm = max_grad_norm
        self.soft_replace_rate = soft_replace_rate
        self.gamma = gamma

        self.explore_noise = explore_noise
        self.explore_noise_decay = explore_noise_decay
        self.explore_decay = explore_decay
        self.explore_noise_min = explore_noise_min

        self.arrival_bonus = arrival_bonus
        self.arrival_tol = arrival_tol

        # environment / device / seeds
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        torch.manual_seed(9)
        if self.device.type == 'cuda':
            torch.cuda.manual_seed(9)
        np.random.seed(9)

        # model name/version & save dir
        self.version = f'GAT-MF'
        self.save_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'model')
        os.makedirs(self.save_dir, exist_ok=True)

        print(f'使用设备: {self.device}')
        print(f'环境设置: {self.num_grid}x{self.num_grid}网格, {self.N}个智能体, {self.num_obstacles}个障碍物')

        # create environment(simulator)
        self.simulator = grid_model.Model(
            grid_size_default=self.num_grid,
            num_agents_default=self.N,
            num_obstacles_default=self.num_obstacles,
            max_steps_default=self.env_max_steps,
            seed_default=self.env_seed,
            fully_connected_adj=self.fully_connected_adj
        )

        # 观测维度
        state_dim = self.simulator.output_dim

        # networks (使用固定的网络架构，不需要参数)
        self.actor = networks.Actor().to(self.device)
        self.actor_target = copy.deepcopy(self.actor).to(self.device)
        self.action_dim = 5  # 从网络定义可知输出维度是5

        self.actor_attention = networks.Attention().to(self.device)
        self.actor_attention_target = copy.deepcopy(self.actor_attention).to(self.device)

        self.critic = networks.Critic().to(self.device)
        self.critic_target = copy.deepcopy(self.critic).to(self.device)

        # 🔧 添加缺失的critic_attention - GAT-MF算法必需
        self.critic_attention = networks.Attention().to(self.device)
        self.critic_attention_target = copy.deepcopy(self.critic_attention).to(self.device)

        self.opt_actor = torch.optim.Adam(params=self.actor.parameters(), lr=self.lr)
        self.opt_critic = torch.optim.Adam(params=self.critic.parameters(), lr=self.lr)
        
        # 🔧 添加注意力网络的优化器
        self.opt_actor_attention = torch.optim.Adam(params=self.actor_attention.parameters(), lr=self.lr)
        self.opt_critic_attention = torch.optim.Adam(params=self.critic_attention.parameters(), lr=self.lr)

        if self.lr_decay:
            self.lr_sche_actor = torch.optim.lr_scheduler.CosineAnnealingLR(self.opt_actor, T_max=self.update_batch, eta_min=0)
            self.lr_sche_critic = torch.optim.lr_scheduler.CosineAnnealingLR(self.opt_critic, T_max=self.update_batch, eta_min=0)
            # 🔧 添加注意力网络的学习率调度器
            self.lr_sche_actor_attention = torch.optim.lr_scheduler.CosineAnnealingLR(self.opt_actor_attention, T_max=self.update_batch, eta_min=0)
            self.lr_sche_critic_attention = torch.optim.lr_scheduler.CosineAnnealingLR(self.opt_critic_attention, T_max=self.update_batch, eta_min=0)

        # generate Adj matrix of graph (已在Model初始化时计算)
        self.Gmat = torch.from_numpy(self.simulator.Gmat).to(self.device)
        
        # 动态邻接支持
        self.use_dynamic_adjacency = getattr(self.simulator, 'use_dynamic_adjacency', False)
        if self.use_dynamic_adjacency:
            print("🔗 训练器启用动态局部连接")
        else:
            print("ℹ️  训练器使用静态邻接矩阵")

        # experience buffer (flat state/action per step)
        self.buffer = Buffer(size=self.buffer_capacity,
                             N=self.N,
                             S_dim=self.N * state_dim,
                             A_dim=self.N * self.action_dim,
                             device=self.device)

        print(f'网络初始化完成 - 状态维度: {state_dim}, 动作维度: {self.action_dim}')
        print('——————————————————————————————————————')

    # ================= helpers =================
    def soft_replace(self, target, source):
        for tar, src in zip(target.parameters(), source.parameters()):
            tar.data.copy_((1.0 - self.soft_replace_rate) * tar.data + self.soft_replace_rate * src.data)

    def get_action(self, action):
        action_vector = F.softmax(action, dim=-1)
        return action_vector

    def get_entropy(self, action):
        weight = F.softmax(action, dim=1)
        action_entropy = -torch.sum(weight * torch.log2(weight))
        return action_entropy

    def get_current_adjacency(self):
        """获取当前时刻的邻接矩阵（张量）"""
        if self.use_dynamic_adjacency and hasattr(self.simulator, 'get_dynamic_adjacency'):
            # 使用动态邻接
            G_tensor = self.simulator.get_dynamic_adjacency(as_tensor=True)
            return G_tensor.to(self.device)
        else:
            # 使用静态邻接
            return self.Gmat
    
    def _reached_flags(self, graph):
        # 使用曼哈顿距离：到达当且仅当 L1 距离为 0（完全重合）
        agent_pos = graph.env_states.agent  # (N, 2) torch.tensor
        goal_pos = graph.env_states.goal    # (N, 2) torch.tensor
        manh = torch.sum(torch.abs(agent_pos - goal_pos), dim=1)
        return manh == 0  # (N,) boolean tensor
    
    def _safety_flags(self, graph):
        """计算每个智能体的安全状态：不与其他智能体或障碍物碰撞（曼哈顿距离≤1 视为不安全）"""
        agent_pos = graph.env_states.agent  # (N, 2) torch.tensor
        N = agent_pos.shape[0]
        
        # 智能体间碰撞检查 (曼哈顿距离 ≤ 1)
        diffs = agent_pos[:, None, :] - agent_pos[None, :, :]  # (N, N, 2)
        manh = torch.sum(torch.abs(diffs), dim=-1)  # (N, N) L1曼哈顿距离
        unsafe_agent_agent = (manh <= 1) & (~torch.eye(N, dtype=torch.bool, device=self.device))
        unsafe_from_agents = torch.any(unsafe_agent_agent, dim=1)  # (N,)
        
        # 智能体与障碍物碰撞检查
        obs_pos = graph.env_states.obstacle.positions  # (num_obstacles, 2)
        if obs_pos.numel() > 0:
            diffs_obs = agent_pos[:, None, :] - obs_pos[None, :, :]  # (N, num_obstacles, 2)
            manh_obs = torch.sum(torch.abs(diffs_obs), dim=-1)  # (N, num_obstacles)
            unsafe_from_obstacles = torch.any(manh_obs <= 1, dim=1)  # (N,)
        else:
            unsafe_from_obstacles = torch.zeros(N, dtype=torch.bool, device=self.device)
        
        # 安全 = 不与其他智能体碰撞 AND 不与障碍物碰撞
        safe = ~(unsafe_from_agents | unsafe_from_obstacles)
        return safe  # (N,) boolean tensor

    # ================= evaluate =================
    def evaluate_policy(self, save_history: bool = False):
        """
        评估阶段：不加噪声；到达即锁定（stay）；与训练的 reached 判定一致。
        """
        total_return = 0.0
        total_arrival = 0.0
        total_safety = 0.0  # 添加安全率统计
        stay_idx = 0  # 若"原地"索引不同，请改

        for _ in range(self.eval_episodes):
            self.simulator.reset()
            current_state = self.simulator.output_record()
            agents_done = torch.zeros(self.N, dtype=torch.bool, device=self.device)
            reached_ever = torch.zeros(self.N, dtype=torch.bool, device=self.device)

            ep_return = 0.0
            ep_safety_steps = 0  # 统计本episode的安全步数
            total_steps = 0      # 统计本episode的总步数

            for _t in range(self.max_steps):
                with torch.no_grad():
                    state = torch.FloatTensor(current_state).to(self.device).unsqueeze(0)  # (1, N, F)
                    
                    # 使用动态邻接矩阵
                    current_adj = self.get_current_adjacency()
                    
                    attn = self.actor_attention(state, current_adj)
                    state_bar = torch.bmm(attn, state)
                    state_all = torch.concat([state, state_bar], dim=-1)
                    logits = self.actor(state_all)
                    probs = self.get_action(logits).squeeze(0)  # (N, A)

                # 锁定已到达
                stay = torch.zeros_like(probs)
                stay[:, stay_idx] = 1.0
                probs = torch.where(agents_done[:, None], stay, probs)

                # 环境推进
                action_vector = probs.detach().cpu().numpy()
                reward_old = self.simulator.get_reward()
                self.simulator.move_miner(action_vector)
                reward_new = self.simulator.get_reward()
                r = float((reward_new - reward_old).mean())
                ep_return += r

                # 更新到达
                graph = self.simulator.graph
                reached_now = self._reached_flags(graph).to(self.device)
                reached_ever = reached_ever | reached_now
                agents_done = agents_done | reached_now
                
                # 计算安全率
                safety_now = self._safety_flags(graph).to(self.device)  # (N,) 当前步安全状态
                ep_safety_steps += float(safety_now.float().mean().item())  # 累积安全步数
                total_steps += 1

                current_state = self.simulator.output_record()
                if bool(agents_done.all().item()):
                    break

            total_return += ep_return
            total_arrival += float(reached_ever.float().mean().item())
            # 计算本episode的平均安全率
            ep_safety_rate = ep_safety_steps / max(1, total_steps)
            total_safety += ep_safety_rate

        avg_return = total_return / max(1, self.eval_episodes)
        avg_arrival = total_arrival / max(1, self.eval_episodes)
        avg_safety = total_safety / max(1, self.eval_episodes)
        print(f"[Eval] return={avg_return:.4f}, arrival={avg_arrival:.4f}, safety={avg_safety:.4f}")
        return avg_return, avg_arrival, avg_safety

    # ================= update =================
    def update_network(self):
        """
        从 buffer 采样更新：
          - target critic/actor 计算 bootstrap 目标 y；
          - 先更新 critic，再更新 actor；
          - 软更新 target。
        """
        if self.buffer.now_size < self.batch_size:
            return

        # 执行多次更新
        for update_step in range(self.update_batch):
            # 每次更新都重新采样
            s_batch, a_batch, r_batch, s1_batch, end_batch = self.buffer.sample(self.batch_size)

            # reshape back to (B, N, *) for networks
            B = s_batch.shape[0]
            state_dim_each = int(s_batch.shape[1] // self.N)
            s_batch = s_batch.view(B, self.N, state_dim_each)
            s1_batch = s1_batch.view(B, self.N, state_dim_each)
            a_batch = a_batch.view(B, self.N, self.action_dim)  # logits or one-hot

            # 🔧 修复：个体化Critic目標计算，使用真正的GAT-MF算法
            with torch.no_grad():
                # 使用当前动态邻接矩阵
                current_adj = self.get_current_adjacency()
                
                # Actor部分：使用actor_attention
                attn_actor_t = self.actor_attention_target(s1_batch, current_adj)
                s1_bar_actor = torch.bmm(attn_actor_t, s1_batch)
                s1_all_actor = torch.concat([s1_batch, s1_bar_actor], dim=-1)
                a1_logits = self.actor_target(s1_all_actor)
                a1 = self.get_action(a1_logits)                 # (B, N, A)
                
                # Critic部分：使用critic_attention
                attn_critic_t = self.critic_attention_target(s1_batch, current_adj)
                s1_bar_critic = torch.bmm(attn_critic_t, s1_batch)
                s1_all_critic = torch.concat([s1_batch, s1_bar_critic], dim=-1)
                a1_bar = torch.bmm(attn_critic_t, a1)           # 对动作也应用注意力
                a1_all = torch.concat([a1, a1_bar], dim=-1)     # 拼接原始动作和注意力增强动作
                q1 = self.critic_target(s1_all_critic, a1_all) # (B, N, 1)
                
                # 个体化目標：r_batch和end_batch现在是(B, N)
                r_batch_expanded = r_batch.unsqueeze(-1)        # (B, N, 1)
                end_batch_expanded = end_batch.unsqueeze(-1)    # (B, N, 1)
                y_individual = r_batch_expanded + self.gamma * q1 * end_batch_expanded  # (B, N, 1)
                y = y_individual.mean(dim=1)                    # (B, 1) 仍然取平均用於loss

            # Critic update - 使用真正的GAT-MF算法
            self.opt_critic.zero_grad()
            self.opt_critic_attention.zero_grad()
            
            # Critic使用自己的attention
            attn_critic = self.critic_attention(s_batch, current_adj)
            s_bar_critic = torch.bmm(attn_critic, s_batch)
            s_all_critic = torch.concat([s_batch, s_bar_critic], dim=-1)
            
            # 对动作也应用critic attention
            a_probs = F.softmax(a_batch, dim=-1)  # 若 a_batch 是 logits，则 softmax 转概率
            a_bar_critic = torch.bmm(attn_critic, a_probs)
            a_all_critic = torch.concat([a_probs, a_bar_critic], dim=-1)
            
            q = self.critic(s_all_critic, a_all_critic)          # (B, N, 1)
            q = q.mean(dim=1)                                    # (B, 1) 对所有智能体取平均
            critic_loss = F.mse_loss(q, y)
            critic_loss.backward()
            
            if self.grad_clip:
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                torch.nn.utils.clip_grad_norm_(self.critic_attention.parameters(), self.max_grad_norm)
            
            self.opt_critic.step()
            self.opt_critic_attention.step()

            # Actor update - 使用真正的GAT-MF算法
            self.opt_actor.zero_grad()
            self.opt_actor_attention.zero_grad()
            
            # Actor使用自己的attention生成动作
            attn_actor = self.actor_attention(s_batch, current_adj)
            s_bar_actor = torch.bmm(attn_actor, s_batch)
            s_all_actor = torch.concat([s_batch, s_bar_actor], dim=-1)
            a_logits = self.actor(s_all_actor)
            a = self.get_action(a_logits)
            
            # 但是Q值评估需要使用critic的attention (重新计算以避免计算图冲突)
            with torch.no_grad():
                attn_critic_new = self.critic_attention(s_batch, current_adj)
                s_bar_critic_new = torch.bmm(attn_critic_new, s_batch)
                s_all_critic_new = torch.concat([s_batch, s_bar_critic_new], dim=-1)
            
            # 对actor生成的动作应用critic attention
            a_bar_critic_new = torch.bmm(attn_critic_new, a)
            a_all_critic_new = torch.concat([a, a_bar_critic_new], dim=-1)
            
            q_actor = self.critic(s_all_critic_new, a_all_critic_new)  # (B, N, 1)
            q_actor = q_actor.mean(dim=1)                              # (B, 1) 对所有智能体取平均
            actor_loss = - q_actor.mean()                              # 标量损失
            actor_loss.backward()
            
            if self.grad_clip:
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                torch.nn.utils.clip_grad_norm_(self.actor_attention.parameters(), self.max_grad_norm)
            
            self.opt_actor.step()
            self.opt_actor_attention.step()

        # 更新完成后进行软更新和学习率调度
        if self.lr_decay:
            self.lr_sche_actor.step()
            self.lr_sche_critic.step()
            self.lr_sche_actor_attention.step()
            self.lr_sche_critic_attention.step()

        # soft update - 包含所有网络
        self.soft_replace(self.actor_target, self.actor)
        self.soft_replace(self.actor_attention_target, self.actor_attention)
        self.soft_replace(self.critic_target, self.critic)
        self.soft_replace(self.critic_attention_target, self.critic_attention)

    # ================= train =================
    def train(self):
        """
        关键：到达即终止（持久），到达只奖一次，已到达持续 end=0；不改变你原有网络与缓冲结构。
        """
        stay_idx = 0
        best_eval = -1e9
        
        # 用于记录训练指标
        episode_returns = []
        arrival_rates = []

        pbar = tqdm(total=self.max_episode, ncols=120)
        for episode in range(self.max_episode):
            # 重置环境
            self.simulator.reset()
            
            # 指数衰减探索噪声（稳定）
            if self.explore_noise_decay:
                self.explore_noise = max(self.explore_noise_min, self.explore_noise * self.explore_decay)

            # 初始状态
            current_state = self.simulator.output_record()  # (N, F)
            agents_done = torch.zeros(self.N, dtype=torch.bool, device=self.device)
            episode_return = 0.0

            for step in range(self.max_steps):
                # 记录 s (flatten to (N*F,))
                state_flat = torch.FloatTensor(current_state.flatten()).to(self.device)
                self.buffer.s[self.buffer.pointer] = state_flat

                with torch.no_grad():
                    state = torch.FloatTensor(current_state).to(self.device).unsqueeze(0)

                    # 使用动态邻接矩阵
                    current_adj = self.get_current_adjacency()
                    
                    Actor_attention = self.actor_attention(state, current_adj)
                    Actor_state_bar = torch.bmm(Actor_attention, state)
                    Actor_state_all = torch.concat([state, Actor_state_bar], dim=-1)
                    action_logits = self.actor(Actor_state_all)   # (1, N, A)

                    # 🔧 修复：探索噪声解耦，使用固定幅度避免初期探索不足
                    if self.explore_noise > 0:
                        noise_scale = 1.0  # 固定噪声幅度，不與logits耦合
                        action_logits = action_logits + torch.randn_like(action_logits) * noise_scale * self.explore_noise

                    probs = self.get_action(action_logits).squeeze(0)  # (N, A)

                # 到达者 -> 强制 stay
                stay = torch.zeros_like(probs)
                stay[:, stay_idx] = 1.0
                probs = torch.where(agents_done[:, None], stay, probs)

                # 环境前进
                action_vector = probs.detach().cpu().numpy()
                reward_old = self.simulator.get_reward()
                self.simulator.move_miner(action_vector)
                reward_new = self.simulator.get_reward()
                reward = reward_new - reward_old                         # numpy (N,1)
                episode_return += float(reward.mean())

                # 到达判定 & 一次性奖励
                current_graph = self.simulator.graph
                reached_now = self._reached_flags(current_graph).to(self.device)   # (N,)
                first_hit = reached_now & (~agents_done)
                
                # 🔧 修复：个体化奖励计算
                individual_rewards = reward.flatten()  # (N,) 個體奖励
                bonus_individual = torch.zeros(self.N, dtype=torch.float32, device=self.device)
                if first_hit.any():
                    bonus_individual[first_hit] = self.arrival_bonus
                    episode_return += float(bonus_individual.sum().item())
                
                total_individual_rewards = individual_rewards + bonus_individual.cpu().numpy()  # (N,)
                episode_return += float(total_individual_rewards.mean() - individual_rewards.mean())  # 只加bonus部分

                # 持久到达掩码更新
                agents_done = agents_done | reached_now

                # 🔧 修复：个体化buffer存儲
                current_state = self.simulator.output_record()
                state_flat_next = torch.FloatTensor(current_state.flatten()).to(self.device)
                self.buffer.s1[self.buffer.pointer] = state_flat_next
                self.buffer.a[self.buffer.pointer] = torch.FloatTensor(action_logits.detach().cpu().numpy().flatten()).to(self.device)
                
                # 存儲个体化奖励 (N,)
                self.buffer.r[self.buffer.pointer] = torch.FloatTensor(total_individual_rewards).to(self.device)

                # 🔧 修复：精細化end_mask - 智能体級別控制
                end_mask = torch.ones(self.N, dtype=torch.float32, device=self.device)
                end_mask[agents_done] = 0.0  # 只對已到达的智能体停止bootstrap
                if step == self.max_steps - 1:
                    end_mask.fill_(0.0)  # 最後一步全部停止
                self.buffer.end[self.buffer.pointer] = end_mask

                # 指针/大小
                self.buffer.now_size = max(self.buffer.now_size, self.buffer.pointer + 1)
                self.buffer.pointer = (self.buffer.pointer + 1) % self.buffer.size

                # 所有 agent 都已到达 → 提前结束本 episode
                if bool(agents_done.all().item()):
                    break

            # 记录当前episode的到达率
            current_arrival = float(agents_done.float().mean().item())
            episode_returns.append(episode_return)
            arrival_rates.append(current_arrival)

            # 更新网络
            if self.buffer.now_size >= self.batch_size and (episode + 1) % self.update_interval == 0:
                try:
                    self.update_network()
                except Exception as e:
                    print(f"\n❌ 网络更新失败 (Episode {episode+1}): {e}")
                    print(f"   - Buffer大小: {self.buffer.now_size}/{self.buffer.size}")
                    print(f"   - Batch大小: {self.batch_size}")
                    raise e

            # 评估/保存
            if self.eval_interval and (episode + 1) % self.eval_interval == 0:
                eval_ret, eval_arrival, eval_safety = self.evaluate_policy()
                if eval_ret > best_eval:
                    best_eval = eval_ret
                    if self.save_interval:
                        self.save_model(episode + 1, eval_ret, eval_arrival)

            # 保存检查点
            if self.save_interval and (episode + 1) % self.save_interval == 0:
                self.save_model(episode + 1, episode_return, current_arrival)

            # 更新进度条
            recent_arrival = np.mean(arrival_rates[-10:]) if len(arrival_rates) >= 10 else np.mean(arrival_rates)
            recent_return = np.mean(episode_returns[-10:]) if len(episode_returns) >= 10 else np.mean(episode_returns)
            pbar.set_postfix({
                'Arr': f'{recent_arrival:.3f}',
                'Ret': f'{recent_return:.2f}',
                'Noise': f'{self.explore_noise:.4f}',
                'Buf': f'{self.buffer.now_size}/{self.buffer.size}'
            })
            pbar.update(1)

        pbar.close()
        final_arrival = np.mean(arrival_rates[-100:]) if len(arrival_rates) >= 100 else np.mean(arrival_rates)
        print(f"\n训练完成！最终到达率: {final_arrival:.3f}")
        return episode_returns, arrival_rates

    def save_model(self, episode, return_val, arrival_rate):
        """保存模型和训练指标"""
        timestamp = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime())
        save_path = os.path.join(self.save_dir, f'{self.num_grid}grid_{timestamp}')
        os.makedirs(save_path, exist_ok=True)
        
        # 保存模型权重 - 包含所有网络
        torch.save(self.actor.state_dict(), os.path.join(save_path, f'actor_{episode}.pth'))
        torch.save(self.actor_target.state_dict(), os.path.join(save_path, f'actor_target_{episode}.pth'))
        torch.save(self.actor_attention.state_dict(), os.path.join(save_path, f'actor_attention_{episode}.pth'))
        torch.save(self.actor_attention_target.state_dict(), os.path.join(save_path, f'actor_attention_target_{episode}.pth'))
        torch.save(self.critic.state_dict(), os.path.join(save_path, f'critic_{episode}.pth'))
        torch.save(self.critic_target.state_dict(), os.path.join(save_path, f'critic_target_{episode}.pth'))
        # 🔧 添加critic_attention的保存
        torch.save(self.critic_attention.state_dict(), os.path.join(save_path, f'critic_attention_{episode}.pth'))
        torch.save(self.critic_attention_target.state_dict(), os.path.join(save_path, f'critic_attention_target_{episode}.pth'))
        
        # 保存配置
        config = {
            'episode': episode,
            'return': float(return_val),
            'arrival_rate': float(arrival_rate),
            'num_grid': self.num_grid,
            'num_agents': self.N,
            'num_obstacles': self.num_obstacles,
            'max_steps': self.max_steps,
            'timestamp': timestamp
        }
        with open(os.path.join(save_path, 'config.json'), 'w') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        print(f"[Save] Episode {episode}: Return={return_val:.4f}, Arrival={arrival_rate:.4f} -> {save_path}")


def main():
    """主训练函数"""
    print("=" * 60)
    print("GAT-MF 多智能体强化学习训练")
    print("=" * 60)
    
    trainer = MARL(
        # —— 环境参数 ——
        num_grid=3,
        num_agents=2,
        num_obstacles=1,
        env_max_steps=50,
        env_seed=123,
        fully_connected_adj=False,  # 改為局部连接

        # —— 训练器采样/优化 ——
        max_steps=80,                    # 增加步数，给更多探索时间
        max_episode=2500,                # 局部连接需要更多训练轮数
        update_batch=16,                 # 降低更新批次，避免过度更新
        batch_size=64,                   # 降低批大小，确保能够更新
        buffer_capacity=200_000,         # 增加缓存容量
        update_interval=1,               # 每个episode都尝试更新
        save_interval=100,
        eval_interval=25,
        eval_episodes=50,                # 增加评估episode数

        # —— 优化/稳定性 ——
        lr=8e-5,                         # 局部连接需要稍高学习率
        lr_decay=True,
        grad_clip=True,
        max_grad_norm=2.0,               # 降低梯度裁剪阈值
        soft_replace_rate=0.008,         # 稍快的target更新适应动态邻接
        gamma=0.88,                      # 稍微提高gamma平衡短期长期奖励

        # —— 探索策略 ——
        explore_noise=0.15,              # 局部连接需要更多探索
        explore_noise_decay=True,
        explore_decay=0.995,             # 更慢的噪声衰减
        explore_noise_min=0.02,          # 保持最小探索噪声

        # —— 奖励/到达阈值 ——
        arrival_bonus=15.0,              # 增加到达奖励
        arrival_tol=1.0,               # 离散网格精确到达判定，只有完全重合才算到达
    )
    
    try:
        episode_returns, arrival_rates = trainer.train()
        print(f"训练成功完成！")
        if len(arrival_rates) > 0:
            print(f"平均到达率: {np.mean(arrival_rates):.3f}")
            if len(arrival_rates) >= 100:
                print(f"最终到达率: {np.mean(arrival_rates[-100:]):.3f}")
            else:
                print(f"最终到达率: {np.mean(arrival_rates[-len(arrival_rates):]):.3f}")
        else:
            print("警告：没有收集到到达率数据")
    except Exception as e:
        print(f"训练过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        print("\n调试信息：")
        print(f"- 检查batch_size({trainer.batch_size})是否过大")
        print(f"- 检查buffer大小({trainer.buffer.now_size})是否足够")
        print(f"- 检查update_interval({trainer.update_interval})设置")


if __name__ == '__main__':
    main()
