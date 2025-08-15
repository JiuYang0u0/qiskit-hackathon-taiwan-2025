import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
from typing import Tuple
class ActorNetwork(nn.Module):
    """
    Actor 網路（策略網路）：
    - 使用 CNN 處理 3D 的電路狀態表示。
    - 輸出三個獨立的機率分佈，分別對應 (gate_type, control_qubit, target_qubit)。
    """
    def __init__(self, state_shape: Tuple[int, int, int], num_gate_types: int, num_qubits: int):
        """
        Args:
            state_shape (Tuple[int, int, int]): 輸入狀態的形狀 (channels, height, width)
                                                i.e., (num_channels, num_qubits, max_moments).
            num_gate_types (int): 閘的種類數量.
            num_qubits (int): 量子位元的數量.
        """
        super().__init__()

        # CNN feature extractor for the 3D state tensor
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels=state_shape[0], out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Calculate the flattened size after the CNN layers
        with torch.no_grad():
            dummy_input = torch.zeros(1, *state_shape)
            flattened_size = self.feature_extractor(dummy_input).shape[1]

        # Shared fully connected layers
        self.shared_layers = nn.Sequential(
            nn.Linear(flattened_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
        )

        # Output heads for each component of the composite action
        self.gate_type_head = nn.Linear(256, num_gate_types)
        self.control_qubit_head = nn.Linear(256, num_qubits)
        self.target_qubit_head = nn.Linear(256, num_qubits)

    def forward(self, state: torch.Tensor) -> Tuple[dist.Categorical, dist.Categorical, dist.Categorical]:
        """
        Forward pass.
        Args:
            state (torch.Tensor): The input state tensor of shape (batch_size, channels, height, width).
        Returns:
            A tuple of three Categorical distributions for gate type, control qubit, and target qubit.
        """
        # Ensure state has a batch dimension
        if state.dim() == 3:
            state = state.unsqueeze(0)

        # --- 防禦性維度重塑 START ---
        if state.dim() == 3:
            state = state.unsqueeze(0)

        n_channels = 6
        if state.shape[1] != n_channels:
            # 如果 Channel 維度不在正確的位置，找到它並移過來
            if state.shape[2] == n_channels:
                state = state.permute(0, 2, 1, 3) # (N, H, C, W) -> (N, C, H, W)
            elif state.shape[3] == n_channels:
                state = state.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)
            else:
                raise ValueError(f"Cannot find channel dimension (size 6) in input shape {state.shape}")
        # --- 防禦性維度重塑 END ---

        features = self.feature_extractor(state)
        shared_output = self.shared_layers(features)

        # Get logits from each head
        gate_logits = self.gate_type_head(shared_output)
        control_logits = self.control_qubit_head(shared_output)
        target_logits = self.target_qubit_head(shared_output)

        # Create categorical distributions
        gate_dist = dist.Categorical(logits=gate_logits)
        control_dist = dist.Categorical(logits=control_logits)
        target_dist = dist.Categorical(logits=target_logits)

        return gate_dist, control_dist, target_dist
class CriticNetwork(nn.Module):
    """
    Critic 網路（價值函數網路）：
    - 用來估計某個狀態 s 的價值 V(s)。
    - 在 PPO 中，Critic 負責提供「狀態價值」來輔助優勢函數的計算。
    """
    def __init__(self, state_dim, fc1_dims=256, fc2_dims=256):
        super().__init__()
        
        # This Critic is designed for a flattened 1D state.
        # To use the same CNN extractor as the Actor, it would need to be redesigned.
        # For now, we assume the state will be flattened before being passed to the critic.
        self.fcnn = nn.Sequential(
                nn.Linear(state_dim, fc1_dims),
                nn.ReLU(),
                nn.Linear(fc1_dims, fc2_dims),
                nn.ReLU(),
                nn.Linear(fc2_dims, 1)
        )
    def forward(self, state):
        """
        Forward pass.
        """
        # print("Runtime state.shape =", state.shape)
        # If state is 3D/4D, flatten it.
        if state.dim() > 1:
            state = state.flatten(start_dim=1)
        value = self.fcnn(state)
        return value