import pickle # To save the qubit operator to Disk.
import os # For file system access.

def save_qubit_op_to_file(qubit_op, file_name='./operators/qubit_op.qpy'):
    base_dir = os.path.dirname(os.path.abspath(__file__)) # Path to src/
    operators_dir = os.path.join(base_dir, '..', 'operators')
    os.makedirs(operators_dir, exist_ok=True)
    file_path = os.path.join(operators_dir, file_name)

    with open(file_path, "wb") as file:
        pickle.dump(qubit_op, file)
    print(f"\nQubit Operator saved to {file_path}!")