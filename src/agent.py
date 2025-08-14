# Importing libraries: 
import os # For file system access.
import torch # For tensor operations.
from torch.optim import Adam, SGD # For gradient update.
from typing import Tuple # For typing annotation.

# Importing custom modules:
from src.actor_critic_networks import ActorNetwork, CriticNetwork
from src.memory import PPOMemory

"""
PPOAgent 是一個策略梯度型強化學習演算法的實作，使用 Proximal Policy Optimization (PPO)。
它主要負責：

建立策略網路（Actor）與價值網路（Critic）

從策略網路抽樣動作

儲存互動資料（state, action, reward, log_prob, value, done）

根據收集到的資料更新 Actor 與 Critic

儲存與載入模型權重
"""
class PPOAgent:
    """
    state_dim：狀態空間的維度，例如一個觀測值是 [x, y, z] 就是 3。
    action_dim：動作空間的大小，例如二元選擇就是 2。
    learning_rate：學習率。
    gamma：折扣因子，控制未來獎勵的權重。
    gae_lambda：GAE（Generalized Advantage Estimation）的 λ，平衡 bias-variance。
    policy_clip：PPO 的策略裁剪參數（限制新舊策略更新幅度）。
    batch_size：每次更新策略的樣本數。
    num_epochs：每批資料用來更新策略的迭代次數。
    optimizer_option：選擇 Adam 或 SGD。
    chkpt_dir：儲存模型檔案的位置。
    """
    '''
    Class for the PPO agent.

    Parameters:
    -----------
    state_dim: int
        Dimension of the state space.
    action_dim: array
        Dimension of the action space.
    learning_rate: float
        Learning rate for the optimizer.
    gamma: float
        Discount factor for future rewards.
    gae_lambda: float
        Lambda parameter for Generalized Advantage Estimation (GAE).
    policy_clip: float
        Clipping parameter for the policy loss.
    batch_size: int
        Batch size for training.
    num_epochs: int
        Number of epochs for training.
    optimizer_option: str
        Choice of optimizer ('Adam' or 'SGD').
    chkpt_dir: str
        Directory to save the model checkpoint.
    '''
    def __init__(self,
                 state_dim,
                 action_dim,
                 learning_rate=0.0003,
                 gamma=0.99,
                 gae_lambda=0.95,
                 policy_clip=0.2,
                 batch_size=64,
                 num_epochs=10,
                 optimizer_option="Adam",
                 chkpt_dir='model/ppo'):

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.policy_clip = policy_clip
        self.num_epochs = num_epochs

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # Instantiate the Actor and Critic networks:
        self.actor = ActorNetwork(state_dim=state_dim, action_dim=action_dim).to(self.device)
        self.critic = CriticNetwork(state_dim=state_dim).to(self.device)

        # Define a dictionary for optimizers:
        optimizers = {
            "Adam": Adam,
            "SGD": SGD
            }
        OptimizerClass = optimizers.get(optimizer_option, Adam) # Default to Adam if not found.
        
        # Define optimizers:
        self.actor_optimizer = OptimizerClass(self.actor.parameters(), lr=learning_rate)
        self.critic_optimizer = OptimizerClass(self.critic.parameters(), lr=learning_rate)

        # Buffer for storing transitions:
        self.memory_buffer = PPOMemory(batch_size)

        # Checkpoint paths:
        self.actor_chkpt = os.path.join(chkpt_dir, 'actor_net_torch_ppo')
        self.critic_chkpt = os.path.join(chkpt_dir, 'critic_net_torch_ppo')
        
    def sample_action(self, observation: torch.tensor) -> Tuple[list, list, float]:
        """
        Sample actions from the policy network given the current state (observation).

        Args:
            observation (torch.tensor): the state representation.

        Returns:
            action (list): list of action(s).
            probs (list): list of probability distribution(s) over action(s).
            value (float): the value from the Critic network.
        """

        '''
        Write your code here.
        '''

        """
        從 policy 中抽樣動作，並返回動作與 log_prob
        """

        action = []
        probs = []
        value = 0.0

        return action, probs, value

    def store_transitions(self, state, action, reward, probs, vals, done):
        """
        This method stores transitions in the memory buffer.
        """
        self.memory_buffer.store_memory(state, action, reward, probs, vals, done)

    def learn(self):
        """
        This method implements the learning step.
        """

        '''
        Write your code here.
        '''

        pass

    def save_models(self):
        print("Saving models...")
        torch.save(self.actor.state_dict(), self.actor_chkpt)
        torch.save(self.critic.state_dict(), self.critic_chkpt)

    def load_models(self):
        print("Loading models...")
        self.actor.load_state_dict(torch.load(self.actor_chkpt))
        self.critic.load_state_dict(torch.load(self.critic_chkpt))