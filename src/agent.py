# Importing libraries: 
import os # For file system access.
import torch # For tensor operations.
from torch.optim import Adam, SGD # For gradient update.
from typing import Tuple # For typing annotation.

# Importing custom modules:
from src.actor_critic_networks import ActorNetwork, CriticNetwork
from src.memory import PPOMemory

class PPOAgent:
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