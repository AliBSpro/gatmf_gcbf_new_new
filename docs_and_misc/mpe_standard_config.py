"""
MPE环境标准化配置文件
用于在不同安全强化学习算法间进行一致的对比实验

Usage:
    from mpe_standard_config import get_mpe_config, create_mpe_env
    
    # 获取配置
    config = get_mpe_config("simple_spread")
    
    # 创建环境 (适配不同框架)
    env = create_mpe_env(config, framework="gym")  # or "gcbf", "custom"
"""

import numpy as np
from typing import Dict, Any, Tuple, Optional


# =============================================================================
# 核心环境参数
# =============================================================================

BASE_CONFIG = {
    # 基础环境设置
    "dt": 0.1,                        # 物理仿真时间步长 (秒)
    "discrete_action": False,          # 使用连续动作空间
    "area_size": 2.0,                 # 世界大小 (2×2米的正方形)
    "world_bounds": [-1.0, 1.0],      # 世界边界 [min, max]
    "dim_p": 2,                       # 位置维度 (x, y)
    "dim_c": 2,                       # 通信维度
    
    # 物理仿真参数
    "integrator": "euler",             # 欧拉积分
    "damping": 0.25,                   # 阻尼系数
    "contact_force": 1e+2,             # 接触力
    "contact_margin": 1e-3,            # 接触边距
}

AGENT_CONFIG = {
    # 智能体物理属性
    "size": 0.05,                     # 智能体半径 (米)
    "mass": 1.0,                      # 质量 (kg)
    "density": 25.0,                  # 材料密度
    "initial_mass": 1.0,              # 初始质量
    
    # 运动参数
    "u_range": 1.0,                   # 控制输入范围 [-1, 1]
    "max_speed": None,                # 最大速度 (无限制)
    "accel": 5.0,                     # 加速度系数
    "movable": True,                  # 可移动
    "collide": True,                  # 可碰撞
    
    # 感知参数
    "silent": False,                  # 可通信
    "blind": False,                   # 可观测
    "adversary": False,               # 非对抗性
    "dummy": False,                   # 非虚拟
    
    # 噪声参数
    "u_noise": None,                  # 动作噪声 (无)
    "c_noise": None,                  # 通信噪声 (无)
}

LANDMARK_CONFIG = {
    # 地标属性
    "size": 0.05,                     # 地标半径 (米)
    "movable": False,                 # 不可移动
    "collide": False,                 # 不可碰撞
    "boundary": False,                # 非边界
    "mass": 1.0,                      # 质量
    "density": 25.0,                  # 密度
}

# =============================================================================
# 安全约束参数
# =============================================================================

SAFETY_CONFIG = {
    # 碰撞避免约束
    "collision_avoidance": {
        "min_distance": 0.18,             # 智能体间最小安全距离 (米)
        "safety_margin": 0.04,            # 额外安全边距 (米)
        "constraint_type": "soft_penalty", # 软约束类型
        "penalty_weight": 8.0,             # 违反惩罚权重
        "barrier_function": "exponential", # 障碍函数类型
        "collision_threshold": 0.1,        # 碰撞检测阈值 (agent.size * 2)
    },
    
    # 边界约束
    "boundary_constraint": {
        "boundary_buffer": 0.08,           # 边界缓冲区 (米)
        "constraint_type": "exponential",  # 渐进式约束
        "strength_factor": 1.5,            # 约束强度因子
        "violation_penalty": 5.0,          # 违反惩罚权重
    }
}

# =============================================================================
# 场景特定参数
# =============================================================================

SCENARIOS = {
    "simple_spread": {
        "scenario_name": "simple_spread",
        "num_agents": 3,
        "num_landmarks": 3,
        "max_episode_length": 25,
        
        # 任务参数
        "task_type": "coverage",
        "success_criteria": "coverage_threshold",
        "coverage_threshold": 0.1,         # 覆盖距离阈值 (米)
        
        # 奖励设计
        "reward_components": {
            "coverage_reward": -1.0,       # 到最近地标距离的负值
            "collision_penalty": -1.0,     # 每次碰撞的惩罚
            "step_penalty": 0.0,           # 时间步惩罚
        },
        
        # 初始配置
        "agent_init_pos": "random",        # 智能体初始位置随机
        "landmark_init_pos": "random",     # 地标初始位置随机
        "init_pos_range": [-1.0, 1.0],    # 初始位置范围
        
        # 观测空间
        "obs_components": [
            "self_velocity",      # [2] 自身速度
            "self_position",      # [2] 自身位置  
            "landmark_rel_pos",   # [2*N] 地标相对位置
            "other_agent_rel_pos" # [2*(M-1)] 其他智能体相对位置
        ],
        
        # 安全约束 (使用标准参数)
        "safety_overrides": {},
    },
    
    "simple_navigation": {
        "scenario_name": "simple_navigation",
        "num_agents": 3,
        "num_landmarks": 3,  # 用作目标点
        "max_episode_length": 25,
        
        # 任务参数
        "task_type": "navigation",
        "success_criteria": "arrival_threshold",
        "arrival_threshold": 0.1,          # 到达距离阈值 (米)
        
        # 奖励设计
        "reward_components": {
            "distance_reward": -1.0,       # 到目标距离的负值
            "collision_penalty": -1.0,     # 每次碰撞的惩罚
            "boundary_penalty": -0.1,      # 边界违反惩罚
        },
        
        # 目标分配
        "target_assignment": "random",     # 目标随机分配
        "target_init_pos": "random",       # 目标位置随机
        
        # 观测空间
        "obs_components": [
            "self_velocity",      # [2] 自身速度
            "self_position",      # [2] 自身位置
            "goal_rel_pos",       # [2] 目标相对位置
            "other_agent_rel_pos" # [2*(M-1)] 其他智能体相对位置
        ],
        
        # 安全约束 (稍微严格一些)
        "safety_overrides": {
            "collision_avoidance": {
                "min_distance": 0.16,     # 稍小的最小距离
            },
            "boundary_constraint": {
                "boundary_buffer": 0.06,  # 稍小的边界缓冲
            }
        },
    }
}

# =============================================================================
# 动作空间配置
# =============================================================================

ACTION_CONFIG = {
    "type": "continuous",
    "shape": 2,                        # [force_x, force_y]
    "env_range": [-5.0, 5.0],         # 环境动作范围 (牛顿)
    "policy_range": [-1.0, 1.0],      # 策略输出范围
    "scaling_factor": 5.0,             # 缩放因子
}

# =============================================================================
# 工具函数
# =============================================================================

def get_mpe_config(scenario: str, custom_params: Optional[Dict] = None) -> Dict[str, Any]:
    """
    获取指定场景的完整MPE配置
    
    Args:
        scenario: 场景名称 ("simple_spread" 或 "simple_navigation")
        custom_params: 自定义参数覆盖
    
    Returns:
        完整的环境配置字典
    """
    if scenario not in SCENARIOS:
        raise ValueError(f"Unknown scenario: {scenario}. Available: {list(SCENARIOS.keys())}")
    
    # 基础配置
    config = {
        **BASE_CONFIG,
        **SCENARIOS[scenario],
        "agent_config": AGENT_CONFIG,
        "landmark_config": LANDMARK_CONFIG,
        "action_config": ACTION_CONFIG,
        "safety_config": SAFETY_CONFIG.copy()
    }
    
    # 应用场景特定的安全参数覆盖
    if "safety_overrides" in config:
        for category, overrides in config["safety_overrides"].items():
            if category in config["safety_config"]:
                config["safety_config"][category].update(overrides)
        del config["safety_overrides"]
    
    # 应用自定义参数覆盖
    if custom_params:
        config = deep_update(config, custom_params)
    
    # 计算派生参数
    config = _compute_derived_params(config)
    
    return config


def _compute_derived_params(config: Dict[str, Any]) -> Dict[str, Any]:
    """计算派生参数"""
    
    # 观测空间维度计算
    num_agents = config["num_agents"]
    num_landmarks = config["num_landmarks"]
    
    if config["scenario_name"] == "simple_spread":
        agent_obs_dim = 4 + 2*(num_agents-1) + 2*num_landmarks  # 自身状态+其他智能体+地标
        global_state_dim = 4*num_agents + 2*num_landmarks
    elif config["scenario_name"] == "simple_navigation":
        agent_obs_dim = 6 + 2*(num_agents-1)  # 自身状态+目标+其他智能体
        global_state_dim = 4*num_agents + 2*num_agents
    else:
        agent_obs_dim = 4 + 2*(num_agents-1)  # 默认
        global_state_dim = 4*num_agents
    
    config["observation_space"] = {
        "agent_obs_dim": agent_obs_dim,
        "global_state_dim": global_state_dim,
        "components": config["obs_components"]
    }
    
    # 安全距离计算
    collision_config = config["safety_config"]["collision_avoidance"]
    total_safe_distance = collision_config["min_distance"] + collision_config["safety_margin"]
    collision_config["total_safe_distance"] = total_safe_distance
    
    # 安全区域计算
    boundary_config = config["safety_config"]["boundary_constraint"]
    world_bounds = config["world_bounds"]
    buffer = boundary_config["boundary_buffer"]
    safe_bounds = [world_bounds[0] + buffer, world_bounds[1] - buffer]
    boundary_config["safe_bounds"] = safe_bounds
    
    return config


def deep_update(base_dict: Dict, update_dict: Dict) -> Dict:
    """深度更新字典"""
    result = base_dict.copy()
    for key, value in update_dict.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = value
    return result


def get_observation_space_info(config: Dict[str, Any]) -> Dict[str, Any]:
    """获取观测空间信息"""
    return {
        "agent_obs_shape": (config["observation_space"]["agent_obs_dim"],),
        "global_state_shape": (config["observation_space"]["global_state_dim"],),
        "action_shape": (config["action_config"]["shape"],),
        "action_range": config["action_config"]["env_range"],
        "num_agents": config["num_agents"],
    }


def get_safety_constraints_info(config: Dict[str, Any]) -> Dict[str, Any]:
    """获取安全约束信息"""
    collision_config = config["safety_config"]["collision_avoidance"]
    boundary_config = config["safety_config"]["boundary_constraint"]
    
    return {
        "collision_avoidance": {
            "min_distance": collision_config["min_distance"],
            "safety_margin": collision_config["safety_margin"],
            "total_safe_distance": collision_config["total_safe_distance"],
            "detection_threshold": collision_config["collision_threshold"],
        },
        "boundary_constraint": {
            "world_bounds": config["world_bounds"],
            "boundary_buffer": boundary_config["boundary_buffer"],
            "safe_bounds": boundary_config["safe_bounds"],
        }
    }


def create_mpe_args_class(config: Dict[str, Any]):
    """创建MPE参数类 (用于on-policy项目)"""
    class MPEArgs:
        def __init__(self):
            self.scenario_name = config["scenario_name"]
            self.num_agents = config["num_agents"] 
            self.num_landmarks = config["num_landmarks"]
            self.max_episode_length = config["max_episode_length"]
            self.discrete_action = config["discrete_action"]
    
    return MPEArgs()


# =============================================================================
# 标准化评估指标
# =============================================================================

def get_evaluation_metrics() -> Dict[str, Any]:
    """获取标准化评估指标定义"""
    return {
        "task_metrics": {
            "simple_spread": [
                "coverage_rate",           # 被覆盖地标数 / 总地标数
                "average_distance",        # 智能体到最近地标的平均距离
                "task_completion_time",    # 完成任务所需时间步数
                "success_rate",            # 成功回合数 / 总回合数
            ],
            "simple_navigation": [
                "arrival_rate",            # 到达目标的智能体数 / 总智能体数
                "average_distance_to_goal",# 智能体到目标的平均距离
                "path_efficiency",         # 直线距离 / 实际路径长度
                "success_rate",            # 所有智能体都到达目标的回合比率
            ]
        },
        
        "safety_metrics": [
            "collision_rate",              # 发生碰撞的回合比率
            "collision_count_per_episode", # 每回合平均碰撞次数
            "boundary_violation_rate",     # 违反边界约束的回合比率
            "min_distance_violations",     # 最小距离违反次数
            "safety_constraint_satisfaction", # 安全约束满足率
            "safety_score",                # 综合安全得分
        ],
        
        "efficiency_metrics": [
            "episode_length",              # 回合长度
            "computational_time",          # 计算时间
            "action_smoothness",           # 动作平滑度
            "energy_consumption",          # 能量消耗 (动作模长积分)
        ]
    }


# =============================================================================
# 使用示例
# =============================================================================

if __name__ == "__main__":
    # 示例: 获取Simple Spread配置
    config = get_mpe_config("simple_spread")
    print("Simple Spread配置:")
    print(f"- 智能体数量: {config['num_agents']}")
    print(f"- 地标数量: {config['num_landmarks']}")
    print(f"- 最大回合长度: {config['max_episode_length']}")
    print(f"- 观测空间维度: {config['observation_space']['agent_obs_dim']}")
    print(f"- 安全距离: {config['safety_config']['collision_avoidance']['total_safe_distance']}")
    
    # 示例: 获取安全约束信息
    safety_info = get_safety_constraints_info(config)
    print(f"\n安全约束:")
    print(f"- 碰撞避免: {safety_info['collision_avoidance']}")
    print(f"- 边界约束: {safety_info['boundary_constraint']}")
    
    # 示例: 自定义参数
    custom_config = get_mpe_config("simple_spread", {
        "num_agents": 5,
        "safety_config": {
            "collision_avoidance": {
                "min_distance": 0.2  # 更严格的安全距离
            }
        }
    })
    print(f"\n自定义配置智能体数量: {custom_config['num_agents']}")
    print(f"自定义安全距离: {custom_config['safety_config']['collision_avoidance']['total_safe_distance']}")
