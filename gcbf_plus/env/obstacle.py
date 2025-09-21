"""
Obstacle classes for GCBF+ environment
障碍物类定义，用于GCBF+环境
"""

from __future__ import annotations
from typing import NamedTuple, Union, List, Tuple
import jax.numpy as jnp
import numpy as np

from ..utils.typing import Array, Pos2d, Pos3d


class Obstacle(NamedTuple):
    """基础障碍物类"""
    center: Array  # 中心位置
    
    
class Rectangle(NamedTuple):
    """2D矩形障碍物"""
    center: Pos2d  # 中心位置 (n_obs, 2)
    size: Array    # 尺寸 (n_obs, 2) - [width, height]
    
    @property
    def points(self) -> Array:
        """获取矩形的四个顶点"""
        # 计算矩形的四个角点
        half_width = self.size[:, 0:1] / 2  # (n_obs, 1)
        half_height = self.size[:, 1:2] / 2  # (n_obs, 1)
        
        # 相对于中心的四个角点
        corners = jnp.array([
            [-1, -1],  # 左下
            [1, -1],   # 右下  
            [1, 1],    # 右上
            [-1, 1]    # 左上
        ])  # (4, 2)
        
        # 缩放并平移到实际位置
        scaled_corners = corners[None, :, :] * jnp.stack([half_width, half_height], axis=-1)[:, None, :]  # (n_obs, 4, 2)
        points = scaled_corners + self.center[:, None, :]  # (n_obs, 4, 2)
        
        return points


class Cuboid(NamedTuple):
    """3D立方体障碍物"""
    center: Pos3d  # 中心位置 (n_obs, 3)
    size: Array    # 尺寸 (n_obs, 3) - [width, height, depth]
    
    @property  
    def points(self) -> Array:
        """获取立方体的8个顶点"""
        half_size = self.size / 2  # (n_obs, 3)
        
        # 立方体的8个角点（相对于中心）
        corners = jnp.array([
            [-1, -1, -1],  # 0: 左下后
            [1, -1, -1],   # 1: 右下后
            [1, 1, -1],    # 2: 右上后
            [-1, 1, -1],   # 3: 左上后
            [-1, -1, 1],   # 4: 左下前
            [1, -1, 1],    # 5: 右下前
            [1, 1, 1],     # 6: 右上前
            [-1, 1, 1]     # 7: 左上前
        ])  # (8, 3)
        
        # 缩放并平移到实际位置
        scaled_corners = corners[None, :, :] * half_size[:, None, :]  # (n_obs, 8, 3)
        points = scaled_corners + self.center[:, None, :]  # (n_obs, 8, 3)
        
        return points


class Sphere(NamedTuple):
    """3D球形障碍物"""
    center: Pos3d  # 中心位置 (n_obs, 3)
    radius: Array  # 半径 (n_obs,)
    
    def contains_point(self, point: Pos3d) -> Array:
        """检查点是否在球内"""
        dist = jnp.linalg.norm(point - self.center, axis=-1)
        return dist <= self.radius


class Circle(NamedTuple):
    """2D圆形障碍物"""
    center: Pos2d  # 中心位置 (n_obs, 2) 
    radius: Array  # 半径 (n_obs,)
    
    def contains_point(self, point: Pos2d) -> Array:
        """检查点是否在圆内"""
        dist = jnp.linalg.norm(point - self.center, axis=-1)
        return dist <= self.radius


# 工具函数
def create_grid_obstacles(grid_size: int, obstacle_positions: List[Tuple[int, int]], 
                         obstacle_size: float = 0.8) -> Rectangle:
    """为网格环境创建矩形障碍物
    
    Args:
        grid_size: 网格大小
        obstacle_positions: 障碍物位置列表 [(row, col), ...]
        obstacle_size: 障碍物相对于网格单元的大小
    
    Returns:
        Rectangle: 矩形障碍物对象
    """
    if not obstacle_positions:
        # 空障碍物
        return Rectangle(
            center=jnp.empty((0, 2)),
            size=jnp.empty((0, 2))
        )
    
    n_obs = len(obstacle_positions)
    centers = jnp.array([[pos[1] + 0.5, grid_size - pos[0] - 0.5] for pos in obstacle_positions])
    sizes = jnp.ones((n_obs, 2)) * obstacle_size
    
    return Rectangle(center=centers, size=sizes)


def create_simple_3x3_obstacle() -> Rectangle:
    """创建3x3网格中的简单障碍物（中心位置）"""
    return create_grid_obstacles(
        grid_size=3,
        obstacle_positions=[(1, 1)],  # 中心位置
        obstacle_size=0.8
    )


# 为了兼容性，创建一些常用的障碍物配置
DEFAULT_3X3_OBSTACLE = create_simple_3x3_obstacle()
