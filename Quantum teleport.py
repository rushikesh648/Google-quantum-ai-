import numpy as np
import cirq

def quantum_teleportation():
    # 1. Allocate three qubits
    q_msg = cirq.NamedQubit("Message (Alice)")
    q_alice = cirq.NamedQubit("Entangled (Alice)")
    q_bob = cirq.NamedQubit("Entangled (Bob)")

    circuit = cirq.Circuit()

    # 2. State Preparation: Prepare an arbitrary quantum state on q_msg
    # Example: Rotate q_msg by arbitrary angles (theta, phi)
    theta = 0.6 * np.pi
    phi = 0.25 * np.pi
    circuit.append([
        cirq.ry(theta)(q_msg),
        cirq.rz(phi)(q_msg)
    ])

    # 3. Create the shared Bell State between Alice and Bob
    circuit.append([
        cirq.H(q_alice),
        cirq.CNOT(q_alice, q_bob)
    ])

    # 4. Alice's Bell-basis measurement on (q_msg, q_alice)
    circuit.append([
        cirq.CNOT(q_msg, q_alice),
        cirq.H(q_msg),
        cirq.measure(q_alice, key="m_alice"),
        cirq.measure(q_msg, key="m_msg")
    ])

    # 5. Bob's conditional corrections using classical control
    # Cirq's `with_classical_controls` executes gates conditioned on measurement keys
    circuit.append([
        cirq.X(q_bob).with_classical_controls("m_alice"),
        cirq.Z(q_bob).with_classical_controls("m_msg")
    ])

    print("--- Quantum Teleportation Circuit ---")
    print(circuit)

    # 6. Verify Teleportation using Density Matrix Simulator
    # To confirm teleportation succeeded without measuring and collapsing Bob's qubit,
    # we inspect the reduced density matrix of Bob's qubit and compare it to the initial message.
    sim = cirq.DensityMatrixSimulator()

    # Get original target state vector of q_msg alone
    prep_circuit = cirq.Circuit(cirq.ry(theta)(q_msg), cirq.rz(phi)(q_msg))
    target_state = cirq.Simulator().simulate(prep_circuit).final_state_vector
    target_rho = np.outer(target_state, np.conj(target_state))

    # Simulate full teleportation circuit
    result = sim.simulate(circuit)
    
    # Extract Bob's reduced density matrix
    bob_rho = cirq.density_matrix_of(result.final_density_matrix, [q_bob])

    print("\n--- Verification ---")
    print(f"Target Density Matrix (Prepared on Message Qubit):\n{np.round(target_rho, 4)}")
    print(f"\nBob's Final Density Matrix (After Teleportation & Correction):\n{np.round(bob_rho, 4)}")

    # Calculate fidelity between initial state and Bob's state
    fidelity = cirq.fidelity(target_rho, bob_rho)
    print(f"\nState Teleportation Fidelity: {fidelity:.6f}")

if __name__ == "__main__":
    quantum_teleportation()
