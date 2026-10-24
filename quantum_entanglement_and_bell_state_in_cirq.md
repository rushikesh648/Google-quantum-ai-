# Hands-On Quantum Computing: Demystifying Entanglement with Google's Cirq

Quantum computing relies on phenomena that set it fundamentally apart from classical computation. The most prominent of these is **quantum entanglement**—a state where two or more particles share physical properties in ways that cannot be described individually.

This guide walks through the conceptual mechanics of quantum entanglement and details how to construct, simulate, and analyze a two-qubit **Bell State** using Google Quantum AI's **Cirq** framework in Python.

---

## 1. Conceptual Foundation: What Is Quantum Entanglement?

In classical systems, bits are strictly independent unless linked by an explicit deterministic rule. A classical register reading `0` on one bit provides zero probabilistic constraint on another bit by default.

In quantum mechanics, a qubit exists in a superposition state represented by a linear combination of basis vectors:

$$|\psi\rangle = \alpha |0\rangle + \beta |1\rangle$$

where $\alpha, \beta \in \mathbb{C}$ and $|\alpha|^2 + |\beta|^2 = 1$.

When two qubits become **entangled**, their composite quantum state cannot be factored into the tensor product of their individual states:

$$|\psi_{AB}\rangle \neq |\psi_A\rangle \otimes |\psi_B\rangle$$

For an entangled Bell pair:

$$|\Phi^+\rangle = \frac{|00\rangle + |11\rangle}{\sqrt{2}}$$

Neither qubit possesses a definite value prior to measurement. However, measuring the first qubit instantly projects the joint state:
* Observing $|0\rangle$ on Qubit A forces Qubit B to collapse into $|0\rangle$.
* Observing $|1\rangle$ on Qubit A forces Qubit B to collapse into $|1\rangle$.

---

## 2. Circuit Blueprint

Creating the Bell state $|\Phi^+\rangle$ from the ground state $|00\rangle$ requires two quantum logic gates:

1. **Hadamard Gate ($H$):** Acts on the control qubit ($q_0$), rotating $|0\rangle$ into an equal superposition:
   $$H|0\rangle = \frac{|0\rangle + |1\rangle}{\sqrt{2}}$$
2. **Controlled-NOT Gate ($CNOT$):** Uses $q_0$ as control and $q_1$ as target. If $q_0$ is $|1\rangle$, it applies an $X$ (bit-flip) gate to $q_1$:
   $$CNOT\left(\frac{|0\rangle + |1\rangle}{\sqrt{2}} \otimes |0\rangle\right) = \frac{|00\rangle + |11\rangle}{\sqrt{2}}$$

### Circuit Schematic

```text
q0: ───[ H ]───●─── [ Measure: alice_result ]
               │
q1: ───────────X─── [ Measure: bob_result ]
```

---

## 3. Python Implementation with Cirq

The following script constructs the circuit, visualizes the schedule of operations, executes a state-vector simulation over 1,000 shots, and aggregates the measurement statistics.

```python
import cirq
from collections import Counter


def build_and_simulate_bell_state():
    # 1. Define named qubits for clear identification
    alice = cirq.NamedQubit("Alice")
    bob = cirq.NamedQubit("Bob")

    # 2. Assemble the quantum circuit
    circuit = cirq.Circuit(
        # Put Alice's qubit into an equal superposition
        cirq.H(alice),
        # Entangle Bob's qubit conditioned on Alice's state
        cirq.CNOT(alice, bob),
        # Measure both qubits to yield classical binary readouts
        cirq.measure(alice, key="alice_result"),
        cirq.measure(bob, key="bob_result"),
    )

    print("=== Circuit Layout ===")
    print(circuit)
    print()

    # 3. Simulate the circuit execution locally
    simulator = cirq.Simulator()
    result = simulator.run(circuit, repetitions=1000)

    # 4. Extract and combine the bitstring results
    measurements = result.histogram(
        key="alice_result,bob_result",
        fold_func=lambda bits: f"{bits[0]}{bits[1]}",
    )

    print("=== Measurement Outcomes (1,000 Repetitions) ===")
    for state, count in sorted(measurements.items()):
        percentage = (count / 1000) * 100
        print(f"State |{state}> : {count:4d} occurrences ({percentage:5.1f}%)")


if __name__ == "__main__":
    build_and_simulate_bell_state()
```

---

## 4. Execution Output & Analysis

Running the script produces output structured as follows:

```text
=== Circuit Layout ===
Alice: ───H───@───M('alice_result')───
              │
Bob: ─────────X───M('bob_result')─────

=== Measurement Outcomes (1,000 Repetitions) ===
State |00> :  494 occurrences ( 49.4%)
State |11> :  506 occurrences ( 50.6%)
```

### Key Technical Observations

* **Zero Orthogonal States:** The states $|01\rangle$ and $|10\rangle$ never register (probability is strictly zero in an ideal, noise-free simulation).
* **Statistical Distribution:** Both $|00\rangle$ and $|11\rangle$ each appear with an empirical probability approaching $p = 0.5$, consistent with the theoretical amplitude squared $\left|\frac{1}{\sqrt{2}}\right|^2 = 0.5$.
* **Application Context:** This entangled pair forms the primitive building block for quantum protocols including Quantum Teleportation, Superdense Coding, and Ekert (E91) Quantum Key Distribution.