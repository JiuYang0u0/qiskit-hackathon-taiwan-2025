import configparser # Config file.
import numpy as np # Numerical operations.
import ast # Convert string to list.
import sys # Command-line arguments.
import os # Directories.
import matplotlib.pyplot as plt # For plotting results

# Helper functions:
#from src.helper_functions.save_qubit_op import save_qubit_op_to_file
from src.helper_functions.load_qubit_op import load_qubit_op_from_file

# Import the agent and environment classes:
from src.agent import PPOAgent
from src.env import VQEnv

##########################################
if __name__ == '__main__':
    #-----------------從命令列取得設定檔名稱------------------------
    # Parse command-line arguments:
    config_file = sys.argv[1]

    #-----------------計算設定檔的完整路徑----------------------
    # Get the path to the config.cfg file:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_file_path = os.path.join(current_dir, config_file)

    #--------------------讀取設定檔----------------------------
    # Load the configuration file:
    config = configparser.ConfigParser()
    config.read(config_file_path)

    #-----------------讀取分子相關參數 (MOL 區段)------------------------
    # Molecule hyperparameters:
    mol_name = config['MOL'].get('mol_name', fallback='Unknown')

    # Atoms:
    atoms_str = config['MOL'].get('atoms', fallback=None)
    atoms = ast.literal_eval(atoms_str) if atoms_str else []

    # Coordinates:
    coordinates_str = config['MOL'].get('coordinates', fallback=None)
    coordinates = ast.literal_eval(coordinates_str) if coordinates_str else ()

    # Number of particles:
    num_particles_str = config['MOL'].get('num_particles', fallback = None)
    num_particles = ast.literal_eval(num_particles_str) if num_particles_str else (0, 0)

    # Multiplicity:
    multiplicity = config.getint('MOL', 'multiplicity', fallback=1)
    # Charge:
    charge = config.getint('MOL', 'charge', fallback=0)
    # Electrons:
    num_electrons = config.getint('MOL', 'num_electrons', fallback = None)
    # Spatial orbitals:
    num_spatial_orbitals = config.getint('MOL', 'num_spatial_orbitals', fallback = None) 
    # Number of qubits:
    num_qubits = config.getint('MOL', 'num_qubits', fallback = None)
    # FCI energy:
    fci_energy = config.getfloat('MOL', 'fci_energy', fallback = None)

    #-----------------讀取訓練相關超參數 (TRAIN 區段)------------------------
    # Convergence tolerance:
    conv_tol = config.getfloat('TRAIN', 'conv_tol', fallback=1e-5)

    # Training hyperparameters:
    learning_rate = config.getfloat('TRAIN', 'learning_rate', fallback=0.0003)
    gamma = config.getfloat('TRAIN', 'gamma', fallback=0.99) 
    gae_lambda = config.getfloat('TRAIN', 'gae_lambda', fallback=0.95) 
    policy_clip = config.getfloat('TRAIN', 'policy_clip', fallback=0.2) 
    batch_size = config.getint('TRAIN', 'batch_size', fallback=64) 
    num_episodes = config.getint('TRAIN', 'num_episodes', fallback=1000) # This is the number of episodes to train the agent.
    num_steps = config.getint('TRAIN', 'num_steps', fallback=20) # This is the number of steps per episode.
    num_epochs = config.getint('TRAIN', 'num_epochs', fallback=10) # This is the number of passes over the same batch of collected data for policy update.
    max_circuit_depth = config.getint('TRAIN', 'max_circuit_depth', fallback=50) 
    conv_tol = config.getfloat('TRAIN', 'conv_tol', fallback=1e-5)
    optimizer_option = config['TRAIN'].get('optimizer_option', fallback='SGD')

    ##########################################
    #---------(可選）建立環境並儲存 qubit operator 的流程------------

    '''
    # Create an instance of the VQEnv class:
    env = VQEnv(molecule_name = "LiH", 
                symbols = atoms, 
                geometry = coordinates, 
                multiplicity = multiplicity, 
                charge = charge,
                num_electrons = num_electrons,
                num_spatial_orbitals = num_spatial_orbitals)

    # Save the qubit operator to disk:
    save_qubit_op_to_file(qubit_op = env.qubit_operator, file_name = "qubit_op_LiH.qpy")
    '''

    #-----------------從檔案載入 qubit operator--------------------
    # Load the qubit operator from disk:
    qubit_operator = load_qubit_op_from_file(file_path = "./src/operators/qubit_op_LiH.qpy")

    ##########################################

    #-----------------建立強化學習環境-------------------------------
    # Create the environment with the loaded qubit operator:
    env = VQEnv(qubit_operator = qubit_operator, 
                num_spatial_orbitals = num_spatial_orbitals, 
                num_particles = num_particles,
                fci_energy = fci_energy)
    
    #-------------------建立 PPO 智能體---------------------------
    # Agent:
    agent = PPOAgent(
        state_shape = env.observation_space.shape, # 傳遞完整的形狀元組
        num_gate_types = 5, # 根據我們的設計，閘的類型是 5 種
        num_qubits = env.num_qubits, # 從環境中獲取量子位元數
        learning_rate = learning_rate,
        gamma = gamma,
        gae_lambda = gae_lambda,
        policy_clip = policy_clip,
        batch_size = batch_size,
        num_epochs = num_epochs,
        optimizer_option = optimizer_option,
        chkpt_dir = 'model/ppo')
    
    #-----------------訓練迴圈--------------------------
    # Create a directory to save results
    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)
    
    final_episode_info = {}
    
    # Training loop:
    for i in range(num_episodes):
        observation, info = env.reset()
        done = False
        score = 0 # Record the total reward for this episode

        while not done:
            # 1. Agent selects an action based on the current state
            action, log_prob, value = agent.sample_action(observation)
            # 2. The environment executes the action
            next_observation, reward, terminated, truncated, info = env.step(action)

            # The episode is done if it's terminated or truncated
            done = terminated or truncated

            # 3. Store the transition in the agent's memory buffer
            agent.store_transitions(observation, action, reward, log_prob, value, done)

            # 4. Update the score and the state
            score += reward
            observation = next_observation

        # 6. After the episode is finished, update the policy
        agent.learn()

        # 7. Log the training progress
        print(f"Episode {i+1}/{num_episodes} | Reward Score: {score:.4f}")

        # 8. Optionally, save the models periodically
        if (i + 1) % 100 == 0:
            agent.save_models()
            
        # Store info from the final episode for plotting
        if i == num_episodes - 1:
            final_episode_info = info

    #-----------------視覺化結果--------------------------
    print("\nVisualizing results of the final episode...")

    # Plot and save the energy curve
    plt.figure(figsize=(10, 5))
    plt.plot(final_episode_info.get('ep_energy', []))
    plt.axhline(y=fci_energy, color='r', linestyle='--', label=f'FCI Energy ({fci_energy:.4f})')
    plt.title('Energy Convergence per Step (Final Episode)')
    plt.xlabel('Step')
    plt.ylabel('Energy')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(results_dir, 'final_episode_energy_curve.png'))
    plt.clf()

    # Plot and save the reward curve
    plt.figure(figsize=(10, 5))
    plt.plot(final_episode_info.get('ep_reward', []))
    plt.title('Reward per Step (Final Episode)')
    plt.xlabel('Step')
    plt.ylabel('Reward')
    plt.grid(True)
    plt.savefig(os.path.join(results_dir, 'final_episode_reward_curve.png'))
    
    print(f"Plots saved to '{results_dir}' directory.")