import pickle # To load the qubit operator from Disk.

def load_qubit_op_from_file(file_path):
    with open(file_path, "rb") as file:
        qubit_op = pickle.load(file)
    print(f"\nQubit Operator loaded from {file_path}!")
    return qubit_op