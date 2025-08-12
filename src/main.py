import configparser # Config file.
import numpy as np # Numerical operations.
import ast # Convert string to list.
import sys # Command-line arguments.
import os # Directories.

# Helper functions:
#from src.helper_functions.save_qubit_op import save_qubit_op_to_file
from src.helper_functions.load_qubit_op import load_qubit_op_from_file

# Import the agent and environment classes:
from src.agent import PPOAgent
from src.env import VQEnv

##########################################
if __name__ == '__main__':
    # Parse command-line arguments:
    config_file = sys.argv[1]

    # Get the path to the config.cfg file:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_file_path = os.path.join(current_dir, config_file)

    # Load the configuration file:
    config = configparser.ConfigParser()
    config.read(config_file_path)

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
    
    # Load the qubit operator from disk:
    qubit_operator = load_qubit_op_from_file(file_path = "./src/operators/qubit_op_LiH.qpy")

    ##########################################

    # Create the environment with the loaded qubit operator:
    env = VQEnv(qubit_operator = qubit_operator, 
                num_spatial_orbitals = num_spatial_orbitals, 
                num_particles = num_particles,
                fci_energy = fci_energy)

    # Agent:
    agent = PPOAgent(
        state_dim = env.observation_space.shape[0],
        action_dim = env.action_space,
        learning_rate = learning_rate,
        gamma = gamma,
        gae_lambda = gae_lambda,
        policy_clip =policy_clip,
        batch_size = batch_size,
        num_epochs = num_epochs,
        optimizer_option = optimizer_option,
        chkpt_dir = 'model/ppo')

    # Training loop:
    for i in range(num_episodes):
        observation = env.reset()
        '''
        Write your code here.
        '''
        break