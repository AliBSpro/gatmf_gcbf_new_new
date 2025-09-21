#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GCBF+ 训练腳本
==============

使用GAT-MF作為參考策略训练GCBF+算法
目標：在保證80%到达率的基礎上提高安全率

环境：3×3网格，2個智能体，1個障碍物
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

# 添加項目路徑
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "grid_gatmf_gcbf+"))


class GCBFPlusTrainer:
    """GCBF+训练器，使用GAT-MF作為參考策略"""
    
    def __init__(self, 
                 gat_mf_params_path: str,
                 output_dir: str = "gcbf_plus_results",
                 verbose: bool = True):
        """
        初始化GCBF+训练器
        
        Args:
            gat_mf_params_path: GAT-MF转换後的参数文件路徑
            output_dir: 输出目錄
            verbose: 是否詳細输出
        """
        self.gat_mf_params_path = Path(gat_mf_params_path)
        self.output_dir = Path(output_dir)
        self.verbose = verbose
        
        # 创建输出目錄
        self.output_dir.mkdir(exist_ok=True)
        (self.output_dir / "models").mkdir(exist_ok=True)
        (self.output_dir / "logs").mkdir(exist_ok=True)
        
        if self.verbose:
            print("🛡️  GCBF+ 训练器初始化")
            print(f"📁 GAT-MF参数: {self.gat_mf_params_path}")
            print(f"📁 输出目錄: {self.output_dir}")
            print("=" * 50)
        
        # 验证参数文件存在
        if not self.gat_mf_params_path.exists():
            raise FileNotFoundError(f"GAT-MF参数文件不存在: {self.gat_mf_params_path}")
    
    def setup_environment_and_algo(self) -> Tuple[Any, Any, Any]:
        """
        设置环境和算法
        
        Returns:
            (env, env_test, algo): 训练环境、测试环境、GCBF+算法
        """
        if self.verbose:
            print("⚙️ 设置环境和算法...")
        
        # 导入必要模塊
        from gcbf_plus.env.gcbf_grid_env import UnifiedGridEnv
        from gcbf_plus.algo.gcbf_plus import GCBFPlus
        from convert.pytorch_to_jax_converter import load_flax_params, inject_uref_to_env
        
        # 加載GAT-MF参数
        if self.verbose:
            print("📥 加載GAT-MF参数...")
        
        flax_params = load_flax_params(str(self.gat_mf_params_path))
        
        if self.verbose:
            print("✅ GAT-MF参数加載成功")
            print(f"   智能体数量: {flax_params.get('n_agents', 'N/A')}")
        
        # 创建环境
        if self.verbose:
            print("🌍 创建环境...")
        
        env = UnifiedGridEnv(
            grid_size=3,
            num_agents=2,
            num_obstacles=1,
            max_steps=60
        )
        
        env_test = UnifiedGridEnv(
            grid_size=3,
            num_agents=2,
            num_obstacles=1,
            max_steps=60
        )
        
        # 注入GAT-MF策略作為u_ref
        if self.verbose:
            print("🔗 注入GAT-MF策略作為參考控制...")
        
        inject_uref_to_env(env, flax_params)
        inject_uref_to_env(env_test, flax_params)
        
        if self.verbose:
            print("✅ 參考控制注入成功")
        
        # 创建GCBF+算法
        if self.verbose:
            print("🤖 创建GCBF+算法...")
        
        algo = GCBFPlus(
            env=env,
            node_dim=3,      # agent/goal/obstacle
            edge_dim=2,      # (dx, dy) 
            state_dim=2,     # (x, y)
            action_dim=2,    # (ux, uy)
            n_agents=2,
            gnn_layers=2,
            batch_size=32,
            buffer_size=10000,
            horizon=16,
            lr_actor=2e-4,              # 提高学习率加速收敛
            lr_cbf=2e-4,              # 同步提高CBF学习率
            alpha=0.8,                # 降低约束强度允许更灵活动作
            eps=0.02,
            inner_epoch=4,
            loss_action_coef=1.2,     # 强化GAT-MF学习
            loss_unsafe_coef=0.6,     # 适度降低不安全权重
            loss_safe_coef=1.0,       # 平衡安全权重
            loss_h_dot_coef=0.6,      # 适度降低CBF导数权重
            max_grad_norm=1.0,
            seed=123
        )
        
        if self.verbose:
            print("✅ GCBF+算法创建成功")
            print(f"   安全约束参数α: {algo.alpha}")
            print(f"   学习率: Actor={algo.lr_actor}, CBF={algo.lr_cbf}")
        
        return env, env_test, algo
    
    def train(self, 
              training_steps: int = 2000,
              eval_interval: int = 50,
              eval_episodes: int = 20,
              save_interval: int = 100,
              n_env_train: int = 8,
              n_env_test: int = 16) -> Dict[str, Any]:
        """
        训练GCBF+算法
        
        Args:
            training_steps: 训练步數
            eval_interval: 评估间隔
            eval_episodes: 评估回合數
            save_interval: 保存间隔
            n_env_train: 並行训练环境數
            n_env_test: 並行测试环境數
            
        Returns:
            训练结果和指標
        """
        if self.verbose:
            print("🚀 开始GCBF+训练...")
            print(f"📊 训练参数:")
            print(f"   - 训练步數: {training_steps}")
            print(f"   - 评估间隔: {eval_interval}")
            print(f"   - 並行环境: {n_env_train} (训练) / {n_env_test} (测试)")
            print()
        
        # 设置环境和算法
        env, env_test, algo = self.setup_environment_and_algo()
        
        # 导入训练器
        from gcbf_plus.trainer.trainer import Trainer
        
        # 训练参数
        training_params = {
            'run_name': f'gcbf_plus_3x3_grid_{int(time.time())}',
            'training_steps': training_steps,
            'eval_interval': eval_interval,
            'eval_epi': eval_episodes,
            'save_interval': save_interval,
        }
        
        # 创建训练器
        trainer = Trainer(
            env=env,
            env_test=env_test,
            algo=algo,
            n_env_train=n_env_train,
            n_env_test=n_env_test,
            log_dir=str(self.output_dir / "logs"),
            seed=123,
            params=training_params,
            save_log=True
        )
        
        if self.verbose:
            print("🎯 开始训练循環...")
        
        # 记录开始时间
        start_time = time.time()
        
        # 开始训练
        try:
            trainer.train()
            training_success = True
        except Exception as e:
            if self.verbose:
                print(f"❌ 训练过程中出現错误: {e}")
            training_success = False
            import traceback
            traceback.print_exc()
        
        # 计算训练时间
        training_time = time.time() - start_time
        
        if self.verbose:
            if training_success:
                print(f"✅ GCBF+训练完成")
            else:
                print(f"⚠️  GCBF+训练遇到问题但已保存部分结果")
            print(f"⏱️  训练用時: {training_time:.1f}秒")
        
        # 嘗試讀取训练结果
        final_metrics = self._load_final_metrics()
        
        # 构建结果
        results = {
            "training_success": training_success,
            "training_time": training_time,
            "final_metrics": final_metrics,
            "config": {
                "training_steps": training_steps,
                "eval_interval": eval_interval,
                "n_env_train": n_env_train,
                "n_env_test": n_env_test,
                "alpha": algo.alpha,
                "lr_actor": algo.lr_actor,
                "lr_cbf": algo.lr_cbf,
            }
        }
        
        # 保存训练结果
        self._save_training_results(results)
        
        # 顯示结果摘要
        if self.verbose:
            self._print_results_summary(results)
        
        return results
    
    def _load_final_metrics(self) -> Dict[str, Any]:
        """加載最終训练指標"""
        try:
            eval_log_path = self.output_dir / "logs" / "eval_metrics.jsonl"
            if eval_log_path.exists():
                with open(eval_log_path, 'r') as f:
                    lines = f.readlines()
                    if lines:
                        # 讀取最後一行作為最終指標
                        final_metrics = json.loads(lines[-1])
                        return final_metrics
            return {}
        except Exception as e:
            if self.verbose:
                print(f"⚠️  無法讀取最終指標: {e}")
            return {}
    
    def _save_training_results(self, results: Dict[str, Any]):
        """保存训练结果"""
        try:
            results_path = self.output_dir / "training_results.json"
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2)
            
            if self.verbose:
                print(f"💾 训练结果已保存: {results_path}")
        except Exception as e:
            if self.verbose:
                print(f"⚠️  保存训练结果失败: {e}")
    
    def _print_results_summary(self, results: Dict[str, Any]):
        """打印结果摘要"""
        print("\n" + "📊" * 20)
        print("📊 GCBF+ 训练结果摘要")
        print("=" * 40)
        
        # 基本信息
        print(f"✅ 训练成功: {'是' if results['training_success'] else '否'}")
        print(f"⏱️  训练用時: {results['training_time']:.1f}秒")
        
        # 最終指標
        final_metrics = results['final_metrics']
        if final_metrics:
            arrival_rate = final_metrics.get('eval/arrival_rate', 'N/A')
            safety_ratio = final_metrics.get('eval/safety_ratio', 'N/A')
            reward = final_metrics.get('eval/reward', 'N/A')
            cost = final_metrics.get('eval/cost', 'N/A')
            
            print(f"📈 最終到达率: {arrival_rate*100:.1f}%" if isinstance(arrival_rate, (int, float)) else f"📈 最終到达率: {arrival_rate}")
            print(f"🛡️  最終安全率: {safety_ratio*100:.1f}%" if isinstance(safety_ratio, (int, float)) else f"🛡️  最終安全率: {safety_ratio}")
            print(f"🎯 最終奖励: {reward:.4f}" if isinstance(reward, (int, float)) else f"🎯 最終奖励: {reward}")
            print(f"💸 最終成本: {cost:.4f}" if isinstance(cost, (int, float)) else f"💸 最終成本: {cost}")
        else:
            print("⚠️  未找到最終指標")
        
        # 配置信息
        config = results['config']
        print(f"\n⚙️ 训练配置:")
        print(f"   - 训练步數: {config['training_steps']}")
        print(f"   - 安全约束α: {config['alpha']}")
        print(f"   - 学习率: Actor={config['lr_actor']}, CBF={config['lr_cbf']}")
        
        print("=" * 40)


def create_example_config() -> Dict[str, Any]:
    """创建示例配置"""
    return {
        "gat_mf_params_path": "results/gat_mf_converted.pkl",
        "output_dir": "gcbf_plus_results",
        "training_params": {
            "training_steps": 500,      # 快速验证模式
            "eval_interval": 25,        # 更频繁评估
            "eval_episodes": 20,
            "save_interval": 100,
            "n_env_train": 8,
            "n_env_test": 16
        }
    }


def main():
    """主函数"""
    print("🛡️  GCBF+ 训练腳本")
    print("使用GAT-MF作為參考策略，提高安全率")
    print("=" * 50)
    
    # 默認配置
    config = create_example_config()
    
    # 检查GAT-MF参数文件
    gat_mf_params_path = Path(config["gat_mf_params_path"])
    if not gat_mf_params_path.exists():
        print(f"❌ GAT-MF参数文件不存在: {gat_mf_params_path}")
        print("請先运行GAT-MF训练並转换模型")
        print("建議运行: python train_full_pipeline.py")
        return False
    
    # 创建训练器
    trainer = GCBFPlusTrainer(
        gat_mf_params_path=str(gat_mf_params_path),
        output_dir=config["output_dir"],
        verbose=True
    )
    
    # 开始训练
    results = trainer.train(**config["training_params"])
    
    # 判断成功與否
    if results["training_success"]:
        print("\n🎉 GCBF+训练成功完成！")
        
        # 检查是否达到目標
        final_metrics = results["final_metrics"]
        if final_metrics:
            arrival_rate = final_metrics.get('eval/arrival_rate', 0)
            safety_ratio = final_metrics.get('eval/safety_ratio', 0)
            
            if isinstance(arrival_rate, (int, float)) and arrival_rate >= 0.75:
                print("✅ 到达率目標达成 (≥75%)")
            
            if isinstance(safety_ratio, (int, float)) and safety_ratio >= 0.9:
                print("✅ 安全率表現优秀 (≥90%)")
        
        return True
    else:
        print("\n❌ GCBF+训练遇到问题")
        return False


if __name__ == "__main__":
    success = main()
    
    if success:
        print(f"\n📁 结果保存在 gcbf_plus_results/ 目錄")
        print("可以使用可视化工具查看训练过程")
    else:
        print("\n請检查错误信息並重試")
