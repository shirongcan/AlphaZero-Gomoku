"""
测试训练的模型是否正常工作（不再乱下）
"""
import numpy as np
from network import PyTorchModel
from games.gomoku import Gomoku
from mcts.new_mcts_alpha import MCTS

def test_model(model_path, n_moves=10):
    """
    测试模型是否只下合法动作
    """
    print(f"测试模型: {model_path}")
    
    # 加载模型
    model = PyTorchModel(board_size=15, action_size=225)
    model.load(model_path)
    
    # 创建MCTS
    mcts = MCTS(
        game_class=Gomoku,
        n_simulations=200,  # 少一点模拟，快速测试
        nn_model=model,
        cpuct=1.0,
        add_dirichlet_noise=False
    )
    
    # 创建游戏
    game = Gomoku(size=15)
    game.current_player = 1
    
    print("\n开始测试...")
    for move_num in range(n_moves):
        print(f"\n第 {move_num + 1} 步:")
        
        # 获取策略
        pi = mcts.run(game, move_num)
        
        # 检查策略
        valid_moves = game.get_valid_moves()
        
        # 1. 检查策略总和
        pi_sum = np.sum(pi)
        print(f"  策略总和: {pi_sum:.6f} (应该接近1.0)")
        
        # 2. 检查是否有非法动作的概率
        illegal_prob = np.sum(pi * (1.0 - valid_moves))
        print(f"  非法动作概率: {illegal_prob:.6f} (应该为0.0)")
        
        # 3. 检查最高概率的动作
        action = np.argmax(pi)
        prob = pi[action]
        r, c = divmod(action, 15)
        is_legal = valid_moves[action] == 1.0
        print(f"  选择动作: ({r}, {c}), 概率: {prob:.4f}, 合法: {is_legal}")
        
        if not is_legal:
            print("  ❌ 错误！模型选择了非法动作！")
            return False
        
        # 执行动作
        game.do_move((r, c))
        game.display()
        
        if game.is_game_over():
            winner = game.get_winner()
            if winner == 0:
                print("\n平局！")
            else:
                print(f"\n玩家 {winner} 获胜！")
            break
    
    print("\n✅ 测试通过！模型只下合法动作。")
    return True

if __name__ == "__main__":
    import os
    import re
    
    model_dir = "models"
    
    # 查找最新的快照模型
    snapshot_files = [f for f in os.listdir(model_dir) 
                     if f.startswith('snapshot_') and f.endswith('.pt')]
    
    if not snapshot_files:
        print("没有找到模型文件！")
        print("训练几个迭代后再测试。")
    else:
        # 按迭代号排序
        def extract_iter_num(filename):
            match = re.search(r"snapshot_iter(\d+)_", filename)
            return int(match.group(1)) if match else -1
        
        snapshot_files.sort(key=extract_iter_num)
        latest_model = os.path.join(model_dir, snapshot_files[-1])
        
        test_model(latest_model, n_moves=10)

