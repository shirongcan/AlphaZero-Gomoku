from typing import Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = F.relu(out)
        return out


class AlphaZeroNet(nn.Module):
    """
    小型AlphaZero风格的残差网络。

    参数:
    - in_channels: 输入通道数（默认为3）
    - board_size: H == W == board_size
    - action_size: 离散动作数量 (board_size * board_size)
    - n_res_blocks: 残差块数量（可调整以平衡速度与性能）
    - channels: 主体中卷积滤波器的数量
    """

    def __init__(self,
                 in_channels: int = 3,
                 board_size: int = 15,
                 action_size: int = 15*15,
                 n_res_blocks: int = 6,
                 channels: int = 128):
        super().__init__()

        self.board_size = board_size
        self.action_size = action_size
        self.channels = channels

        # 初始卷积块
        self.conv = nn.Conv2d(in_channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(channels)

        # 残差塔
        self.res_blocks = nn.ModuleList([ResidualBlock(channels) for _ in range(n_res_blocks)])

        # 策略头
        # 小卷积 -> 展平 -> 线性层到action_size
        self.policy_conv = nn.Conv2d(channels, 2, kernel_size=1, bias=False)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * board_size * board_size, action_size)

        # 价值头
        # 小卷积 -> 展平 -> 两层MLP -> 标量
        self.value_conv = nn.Conv2d(channels, 1, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(board_size * board_size, 64)
        self.value_fc2 = nn.Linear(64, 1)

        self._init_weights()

    def _init_weights(self):
        # 对卷积层使用Kaiming初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        输入:
            x: (B, in_channels, H, W) float32
        返回:
            policy_logits: (B, action_size)  (原始logits，在softmax之前)
            value: (B, 1) float 在tanh之后的值，范围[-1,1]
        """
        # 主体
        out = self.conv(x)
        out = self.bn(out)
        out = F.relu(out)

        for blk in self.res_blocks:
            out = blk(out)

        # 策略头
        p = self.policy_conv(out)
        p = self.policy_bn(p)
        p = F.relu(p)
        p = p.view(p.shape[0], -1)
        policy_logits = self.policy_fc(p)  # (B, action_size)

        # 价值头
        v = self.value_conv(out)
        v = self.value_bn(v)
        v = F.relu(v)
        v = v.view(v.shape[0], -1)
        v = F.relu(self.value_fc1(v))
        v = self.value_fc2(v)
        value = torch.tanh(v)

        return policy_logits, value
    
    def predict(self, state):
        """
        预测方法，接收状态并返回策略和价值。
        """
        # 将状态（如果仍是numpy数组）转换为PyTorch张量
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)  # 添加额外的维度用于批次大小(1)

        # 将张量传入网络
        policy_logits, value = self(state_tensor)

        return policy_logits, value


class PyTorchModel:
    """
    围绕AlphaZeroNet的包装类，提供:
     - predict(encoded_states) -> (policy_np, value_np)
     - predict_batch(list_of_encoded_states) -> (policy_np, value_np)
     - train_batch(...) -> 损失字典和反向传播步骤
     - save / load
    """

    def __init__(self,
                 board_size: int = 15,
                 action_size: Optional[int] = None,
                 device: Optional[str] = None,
                 n_res_blocks: int = 3,
                 channels: int = 64,
                 lr: float = 1e-3,
                 weight_decay: float = 1e-4):
        self.board_size = board_size
        self.action_size = action_size if action_size is not None else board_size * board_size

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.net = AlphaZeroNet(
            in_channels=3,
            board_size=board_size,
            action_size=self.action_size,
            n_res_blocks=n_res_blocks,
            channels=channels
        ).to(self.device)

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr, weight_decay=weight_decay)
        self.value_loss_fn = nn.MSELoss()
        self.policy_loss_fn = nn.KLDivLoss(reduction='batchmean')  # log_probs vs target probs

    # -------------------------
    # 预测（用于MCTS的批次）
    # -------------------------
    def predict(self, encoded_states: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        encoded_states: numpy数组 (B, C, H, W) float32
        返回:
            policy_probs: np.array (B, action_size) float32
            values: np.array (B, 1) float32
        """
        training = self.net.training
        self.net.eval()
        with torch.no_grad():
            x = torch.from_numpy(encoded_states.astype(np.float32)).to(self.device)
            logits, value = self.net(x)
            probs = F.softmax(logits, dim=1).cpu().numpy()
            values = value.cpu().numpy()
        self.net.train(training)
        return probs, values

    # -------------------------
    # 便利方法：从列表创建批次
    # -------------------------
    def predict_batch(self, states_list: list) -> Tuple[np.ndarray, np.ndarray]:
        """
        states_list: np数组列表 (C,H,W)
        返回: policy_probs, values 作为np数组
        """
        batch = self.make_batch_from_states(states_list)
        return self.predict(batch)

    # -------------------------
    # 工具方法：从状态编码推断合法动作
    # -------------------------
    @staticmethod
    def get_valid_moves_from_state(state: np.ndarray) -> np.ndarray:
        """
        从状态编码推断合法动作。
        状态编码格式: [C, H, W]，其中C=3
        - 通道0: 当前玩家的棋子
        - 通道1: 对手的棋子
        - 通道2: turn信息
        如果某个位置的通道0和通道1都是0，则该位置是空的（合法动作）。
        """
        # state shape: (C, H, W) 或 (B, C, H, W)
        if len(state.shape) == 3:
            state = state[np.newaxis, ...]  # 添加batch维度
        
        batch_size, channels, height, width = state.shape
        action_size = height * width
        
        # 获取每个位置的占用情况：通道0或通道1有值的位置被占用
        occupied = (state[:, 0, :, :] > 0.5) | (state[:, 1, :, :] > 0.5)  # (B, H, W)
        
        # 合法动作：未被占用的位置
        valid_moves = (~occupied).astype(np.float32)  # (B, H, W)
        valid_moves = valid_moves.reshape(batch_size, action_size)  # (B, action_size)
        
        if len(state.shape) == 3:
            return valid_moves[0]  # 移除batch维度
        return valid_moves

    # -------------------------
    # 单个训练批次
    # -------------------------
    def train_batch(self,
                    states: np.ndarray,
                    target_pis: np.ndarray,
                    target_vs: np.ndarray,
                    epochs: int = 1) -> dict:
        self.net.train()
        states_t = torch.from_numpy(states.astype(np.float32)).to(self.device)
        target_pis_t = torch.from_numpy(target_pis.astype(np.float32)).to(self.device)
        target_vs_t = torch.from_numpy(target_vs.astype(np.float32)).to(self.device)

        # 从状态编码推断合法动作掩码
        valid_moves = self.get_valid_moves_from_state(states)  # (B, action_size)
        valid_moves_t = torch.from_numpy(valid_moves.astype(np.float32)).to(self.device)
        
        # 确保目标策略只包含合法动作（归一化）
        target_pis_t = target_pis_t * valid_moves_t
        target_pis_t = target_pis_t / (target_pis_t.sum(dim=1, keepdim=True) + 1e-8)

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_loss = 0.0

        for _ in range(epochs):
            self.optimizer.zero_grad()
            logits, values = self.net(states_t)
            
            # 对logits应用掩码：将非法动作的logits设为负无穷大
            masked_logits = logits.clone()
            masked_logits = masked_logits - (1.0 - valid_moves_t) * 1e9
            
            log_probs = F.log_softmax(masked_logits, dim=1)

            policy_loss = self.policy_loss_fn(log_probs, target_pis_t)
            value_loss = self.value_loss_fn(values, target_vs_t)

            loss = policy_loss + value_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), 3.0)
            self.optimizer.step()

            total_policy_loss += float(policy_loss.item())
            total_value_loss += float(value_loss.item())
            total_loss += float(loss.item())

        ne = float(epochs)
        return {
            "policy_loss": total_policy_loss / ne,
            "value_loss": total_value_loss / ne,
            "total_loss": total_loss / ne
        }

    # -------------------------
    # 保存 / 加载
    # -------------------------
    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        state = {
            "net": self.net.state_dict(),
            "opt": self.optimizer.state_dict(),
            "board_size": self.board_size,
            "action_size": self.action_size
        }
        torch.save(state, path)

    def load(self, path: str, map_location: Optional[str] = None) -> None:
        map_location = map_location or self.device
        state = torch.load(path, map_location=map_location)
        self.net.load_state_dict(state["net"])
        if "opt" in state and state["opt"] is not None:
            try:
                self.optimizer.load_state_dict(state["opt"])
            except Exception:
                pass

    # -------------------------
    # 工具方法：将列表转换为批次
    # -------------------------
    @staticmethod
    def make_batch_from_states(list_of_encoded_states: list) -> np.ndarray:
        return np.stack(list_of_encoded_states, axis=0).astype(np.float32)