from typing import List, Tuple
from qiskit.circuit import QuantumCircuit, Parameter

def decode_actions_into_circuit(actions: List[Tuple[int, int, int]], num_qubits: int) -> QuantumCircuit:
    """
    Decode the actions taken by the agent into quantum circuit operations.
    
    Args:
        actions (List[Tuple[int, int, int]]):
            A list of actions taken by the agent.
            Each action is a tuple: (gate_type, target_qubit, control_qubit).
            For single-qubit gates, control_qubit can be ignored (e.g., -1 or None).
        num_qubits (int): The number of qubits in the quantum circuit.

    Returns:
        qc (QuantumCircuit): A quantum circuit constructed based on the agent's actions.
    """
    qc = QuantumCircuit(num_qubits)

    # 0: Rx, 1: Ry, 2: Rz, 3: H, 4: CNOT
    gate_map = {0: 'rx', 1: 'ry', 2: 'rz', 3: 'h', 4: 'cnot'}

    param_counter = 0

    for action in actions:
        gate_type_idx, target_qubit, control_qubit = action
        gate_name = gate_map.get(gate_type_idx)

        if gate_name is None:
            # Skip if gate_type_idx is invalid
            continue

        # Ensure qubit indices are within bounds
        if not (0 <= target_qubit < num_qubits and (control_qubit is None or -1 <= control_qubit < num_qubits)):
            continue

        if gate_name in ['rx', 'ry', 'rz']:
            # For rotation gates, create a unique variational parameter
            theta = Parameter(f'θ_{param_counter}')
            param_counter += 1
            getattr(qc, gate_name)(theta, target_qubit)

        elif gate_name == 'h':
            qc.h(target_qubit)

        elif gate_name == 'cnot':
            # Ensure control_qubit is valid for CNOT
            if control_qubit is not None and control_qubit != -1 and control_qubit != target_qubit:
                qc.cx(control_qubit, target_qubit)

    return qc