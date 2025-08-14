import torch 
import numpy as np 
from qiskit.circuit import QuantumCircuit 
def encode_circuit_into_input_embedding(qc: QuantumCircuit, max_moments: int) -> torch.Tensor:     
    """
    Encodes a QuantumCircuit into a 3D tensor representation.
    (Encode (convert) the quantum circuit into a tensor representation for input to the agent.)
    
    The tensor shape is (num_qubits, max_moments, num_channels).
    - num_channels is 6, based on the gate set {Rx, Ry, Rz, H, CNOT}.

    Args:
        qc (QuantumCircuit): The quantum circuit to encode.
        max_moments (int): The depth of the output tensor.

    Returns:
        torch.Tensor: The 3D tensor representation of the circuit.
    """
    num_qubits = qc.num_qubits
    n_channels = 6

    # Channel mapping based on the chosen gate set
    gate_to_channel = {
        'rx': 0,
        'ry': 1,
        'rz': 2,
        'h': 3,
        'cx_control': 4,  # Use 'cx' for CNOT as it's a standard name in Qiskit
        'cx_target': 5
    }

    # Initialize the tensor to all zeros
    circuit_tensor = torch.zeros((num_qubits, max_moments, n_channels), dtype=torch.float32)

    # Array to track the next available moment for each qubit
    qubit_moment_counters = np.zeros(num_qubits, dtype=int)

    # Iterate over each instruction in the circuit data
    for instruction in qc.data:
        gate = instruction.operation
        # Standardize gate name (e.g., 'cx' is often used for CNOT)
        gate_name = gate.name.lower()
        if gate_name == 'cnot':
            gate_name = 'cx'

        qubit_indices = [q.index for q in instruction.qubits]

        # Determine the moment for the current gate
        # The gate can only be placed after all its qubits are free
        moment = 0
        if qubit_indices:
            moment = int(np.max(qubit_moment_counters[qubit_indices]))

        # If the circuit is deeper than max_moments, we cannot place the gate
        if moment >= max_moments:
            print(f"Warning: Circuit depth exceeds max_moments ({max_moments}). "
                  f"Gate {gate_name} at moment {moment} will be ignored.")
            continue

        # --- Update tensor based on gate type ---
        if gate_name in ['rx', 'ry', 'rz', 'h']:
            qubit_idx = qubit_indices[0]
            channel_idx = gate_to_channel.get(gate_name)
            if channel_idx is not None:
                circuit_tensor[qubit_idx, moment, channel_idx] = 1.0

        elif gate_name == 'cx':
            control_qubit_idx = qubit_indices[0]
            target_qubit_idx = qubit_indices[1]

            # Set the control and target channels
            control_channel = gate_to_channel['cx_control']
            target_channel = gate_to_channel['cx_target']

            circuit_tensor[control_qubit_idx, moment, control_channel] = 1.0
            circuit_tensor[target_qubit_idx, moment, target_channel] = 1.0

        # Update the moment counters for all involved qubits
        # They will be free at the next moment
        for qubit_idx in qubit_indices:
            qubit_moment_counters[qubit_idx] = moment + 1

    return circuit_tensor