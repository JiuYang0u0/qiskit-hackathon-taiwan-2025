# Importing libraries: 
import os # For file system access.
import torch # For tensor operations.
from torch.optim import Adam, SGD # For gradient update.
from typing import Tuple # For typing annotation.

# Importing custom modules:
from src.actor_critic_networks import ActorNetwork, CriticNetwork
from src.memory import PPOMemory
import numpy as np

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
                 # 將 state_dim 改為 state_shape
                 state_shape: Tuple[int, int, int],
                 # 將 action_dim 改為更具體的參數
                 num_gate_types: int,
                 num_qubits: int,
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
        # self.actor = ActorNetwork(state_dim=state_dim, action_dim=action_dim).to(self.device)
        self.actor = ActorNetwork(
                state_shape=state_shape,
                num_gate_types=num_gate_types,
                num_qubits=num_qubits
            ).to(self.device)
        # state_shape is (C, H, W)
        critic_state_dim = state_shape[0] * state_shape[1] * state_shape[2]
        self.critic = CriticNetwork(state_dim=critic_state_dim).to(self.device)

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
        
    def sample_action(self, observation: np.ndarray) -> Tuple[list, list, float]:
        """
        Sample a composite action from the policy network given the current state.

        Args:
            observation (np.ndarray): The state representation, expected to be convertible to a 3D tensor.
        
        Returns:
            action (list): A list containing the sampled [gate_type, control_qubit, target_qubit].
            log_probs (list): A list of log probabilities for each part of the action.
            value (float): The value of the state from the Critic network.
        """
        # Convert observation to a tensor and send to the correct device
        # The observation from env is likely a numpy array, so we convert it.
        state = torch.tensor(observation, dtype=torch.float32).to(self.device)
        
        # Ensure the state has the correct 3D shape for the CNN, and add a batch dimension
        if state.dim() == 2: # Assuming (H, W) -> need to add Channel dimension
                state = state.unsqueeze(0) # Add channel dimension: (1, H, W)
        if state.dim() == 3: # Assuming (C, H, W)
            state = state.unsqueeze(0) # Add batch dimension: (1, C, H, W)
            # We are not training here, so we don't need to track gradients
        
        with torch.no_grad():
            # 1. Get action distributions from the Actor network
            # print("Init state_shape =", state.shape)
            gate_dist, control_dist, target_dist = self.actor(state)

            # 2. Get state value from the Critic network
            # The critic network handles flattening internally
            value = self.critic(state)

            # 3. Sample an action from each distribution
            gate_type = gate_dist.sample()
            control_qubit = control_dist.sample()
            target_qubit = target_dist.sample()

            # 4. Calculate the log probability of each sampled action
            gate_log_prob = gate_dist.log_prob(gate_type)
            control_log_prob = control_dist.log_prob(control_qubit)
            target_log_prob = target_dist.log_prob(target_qubit)
        
        # Combine actions and log_probs into lists
        action = [gate_type.item(), control_qubit.item(), target_qubit.item()]
        log_probs = [gate_log_prob.item(), control_log_prob.item(), target_log_prob.item()]

        return action, log_probs, value.item()

    def store_transitions(self, state, action, reward, probs, vals, done):
        """
        This method stores transitions in the memory buffer.
        """
        if isinstance(state, torch.Tensor):
            state = state.cpu().numpy()
        self.memory_buffer.store_memory(state, action, reward, probs, vals, done)

    def learn(self):
        """
        This method implements the PPO learning step.
        It computes advantages, and updates the actor and critic networks.
        """
        for _ in range(self.num_epochs):
            # 1. Generate batches from memory
            state_arr, action_arr, old_prob_arr, vals_arr, \
            reward_arr, dones_arr, batches = \
                    self.memory_buffer.generate_batches()
            
            # 2. Calculate advantages using GAE
            advantage = np.zeros(len(reward_arr), dtype=np.float32)
            for t in range(len(reward_arr)-1, -1, -1):
                discount = 1
                a_t = 0
                for k in range(t, len(reward_arr)-1):
                       a_t += discount * (reward_arr[k] + self.gamma * vals_arr[k+1] * \
                            (1-int(dones_arr[k])) - vals_arr[k])
                       discount *= self.gamma * self.gae_lambda
                advantage[t] = a_t

            advantage = torch.tensor(advantage).to(self.device)
            values = torch.tensor(vals_arr).to(self.device)

            # 3. Iterate over mini-batches
            for batch in batches:
                states = torch.tensor(state_arr[batch], dtype=torch.float32).to(self.device)
                old_probs = torch.tensor(old_prob_arr[batch], dtype=torch.float32).to(self.device)
                actions = torch.tensor(action_arr[batch], dtype=torch.int64).to(self.device)

                # 4. Re-evaluate actions with the current policy
                gate_dist, control_dist, target_dist = self.actor(states)
                critic_value = self.critic(states).squeeze()

                # Calculate new log probabilities for the composite action
                new_gate_log_prob = gate_dist.log_prob(actions[:, 0])
                new_control_log_prob = control_dist.log_prob(actions[:, 1])
                new_target_log_prob = target_dist.log_prob(actions[:, 2])

                # Sum the log_probs for the composite action
                new_probs = new_gate_log_prob + new_control_log_prob + new_target_log_prob

                # Old probs were stored as a list of 3, sum them up
                old_probs = old_probs.sum(axis=1)

                # 5. Calculate Policy Loss (L_CLIP)
                prob_ratio = (new_probs - old_probs).exp()
                weighted_probs = advantage[batch] * prob_ratio
                weighted_clipped_probs = torch.clamp(prob_ratio, 1-self.policy_clip, 1+self.policy_clip) * advantage[batch]
                
                actor_loss = -torch.min(weighted_probs, weighted_clipped_probs).mean()

                # 6. Calculate Value Loss
                returns = advantage[batch] + values[batch]
                critic_loss = (returns - critic_value)**2
                critic_loss = critic_loss.mean()

                # 7. Calculate Total Loss and perform backpropagation
                total_loss = actor_loss + 0.5*critic_loss

                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                total_loss.backward()
                self.actor_optimizer.step()
                self.critic_optimizer.step()

        # 8. Clear memory for the next round of data collection
        self.memory_buffer.clear_memory()

    def save_models(self):
        print("Saving models...")
        torch.save(self.actor.state_dict(), self.actor_chkpt)
        torch.save(self.critic.state_dict(), self.critic_chkpt)

    def load_models(self):
        print("Loading models...")
        self.actor.load_state_dict(torch.load(self.actor_chkpt))
        self.critic.load_state_dict(torch.load(self.critic_chkpt))