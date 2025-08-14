import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist

class ActorNetwork(nn.Module):
    """
    Actor 網路（策略網路）：
    - 用來輸出在當前狀態下採取各種動作的機率分佈。
    - 在 PPO 中，Actor 負責學習 policy π(a|s)，也就是「在狀態 s 下，選擇動作 a 的機率」。
    """
    """
    Actor network for the PPO agent.

    Parameters:
    -----------
    state_dim: int
        Dimension of the state space.
    action_dim: array
        Dimension of the action space.
    """
    def __init__(self, state_dim, action_dim):
        # Run the constructor of the parent class (nn.Module):
        super().__init__()

        '''
        Write your code here.
        '''

        num_gate_types = action_dim
        
        

        # # Example:
        # num_gate_types = action_dim  # 動作種類數量（例如量子閘的種類）
        # # 第一層全連接層，將輸入狀態映射到 128 維特徵
        # self.fc1 = nn.Linear(state_dim, 128)
        # # 第二層全連接層，進一步提取特徵
        # self.fc2 = nn.Linear(128, 128)
        # # 輸出層，對應到每個動作的 logit（尚未經過 softmax）
        # self.output_layer = nn.Linear(128, num_gate_types)

    def forward(self, state):
        """
        Forward pass.
        """

        '''
        Write your code here.
        '''
        
        # Example:
        # 隱藏層使用 ReLU 激活函數
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        # 計算每個動作的 logit（未經 softmax）
        # Gate Type Distribution:
        gate_logits = self.output_layer(x) # Raw logits.
        # 將 logit 轉為機率分佈（Categorical 會自動套用 softmax）
        gate_dist = dist.Categorical(logits=gate_logits) # Categorical internally applies Softmax.
        return gate_dist

class CriticNetwork(nn.Module):
    """
    Critic 網路（價值函數網路）：
    - 用來估計某個狀態 s 的價值 V(s)。
    - 在 PPO 中，Critic 負責提供「狀態價值」來輔助優勢函數的計算。
    """
    """
    Critic network for the PPO agent.

    Parameters:
    -----------
    state_dim: int
        Dimension of the state space.
    fc1_dims: int
        Number of neurons in the first hidden layer.
    fc2_dims: int
        Number of neurons in the second hidden layer.
    """
    def __init__(self, state_dim, fc1_dims=256, fc2_dims=256):
        # Run the constructor of the parent class (nn.Module):
        super().__init__()

        # Neural network layers:
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
        value = self.fcnn(state)
        return value
