import numpy as np
import sympy
import cirq
from scipy.optimize import minimize
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report

# ==========================================
# 1. Dataset Generation & Preprocessing
# ==========================================
# Generate binary classification moon dataset
X_raw, y = make_moons(n_samples=120, noise=0.15, random_state=42)

# Scale features into [0, pi] for angle encoding (Ry rotations)
scaler = MinMaxScaler(feature_range=(0, np.pi))
X = scaler.fit_transform(X_raw)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# ==========================================
# 2. Quantum Circuit Architecture (Cirq)
# ==========================================
num_qubits = 2
qubits = cirq.LineQubit.range(num_qubits)

# Define symbolic parameters
x_symbols = [sympy.Symbol(f"x_{i}") for i in range(num_qubits)]
theta_symbols = [sympy.Symbol(f"theta_{i}") for i in range(4)]  # 4 trainable weights

circuit = cirq.Circuit()

# (a) Feature Encoding Layer: Angle encoding via Ry
for i in range(num_qubits):
    circuit.append(cirq.ry(x_symbols[i])(qubits[i]))

# (b) Variational Ansatz: Rotations + Entanglement (CNOT)
circuit.append(cirq.ry(theta_symbols[0])(qubits[0]))
circuit.append(cirq.ry(theta_symbols[1])(qubits[1]))
circuit.append(cirq.CNOT(qubits[0], qubits[1]))
circuit.append(cirq.rz(theta_symbols[2])(qubits[0]))
circuit.append(cirq.rz(theta_symbols[3])(qubits[1]))

print("=== VQC Circuit Architecture ===")
print(circuit)

# ==========================================
# 3. Expectation Value & VQC Model Class
# ==========================================
class CirqVQC:
    def __init__(self, circuit, qubits, x_symbols, theta_symbols):
        self.circuit = circuit
        self.qubits = qubits
        self.x_symbols = x_symbols
        self.theta_symbols = theta_symbols
        self.simulator = cirq.Simulator()
        self.weights = None
        
        # Observable: Pauli-Z on target readout qubit (qubits[1])
        # Returns expectation <Z> in [-1, +1]
        self.observable = cirq.Z(self.qubits[1])

    def _predict_single(self, x_sample, weights):
        param_dict = {sym: val for sym, val in zip(self.x_symbols, x_sample)}
        param_dict.update({sym: val for sym, val in zip(self.theta_symbols, weights)})
        
        resolver = cirq.ParamResolver(param_dict)
        resolved_circuit = cirq.resolve_parameters(self.circuit, resolver)
        
        # Exact state vector expectation value
        result = self.simulator.simulate(resolved_circuit)
        exp_z = self.observable.expectation_from_state_vector(
            result.final_state_vector,
            qubit_map={q: i for i, q in enumerate(self.qubits)}
        ).real
        
        # Map <Z> in [-1, 1] to class probability P(y=1) in [0, 1]
        prob = (1.0 - exp_z) / 2.0
        return np.clip(prob, 1e-7, 1 - 1e-7)

    def predict_proba(self, X, weights=None):
        if weights is None:
            weights = self.weights
        return np.array([self._predict_single(x, weights) for x in X])

    def predict(self, X, weights=None):
        probs = self.predict_proba(X, weights)
        return (probs >= 0.5).astype(int)

    def _loss(self, weights, X, y):
        probs = self.predict_proba(X, weights)
        # Binary cross-entropy loss
        bce = -np.mean(y * np.log(probs) + (1 - y) * np.log(1 - probs))
        return bce

    def fit(self, X, y, maxiter=25):
        # Initialize small random rotation weights
        np.random.seed(42)
        initial_weights = np.random.uniform(0, 2 * np.pi, size=len(self.theta_symbols))
        
        print("\nStarting Classical Optimization (COBYLA)...")
        opt_res = minimize(
            fun=self._loss,
            x0=initial_weights,
            args=(X, y),
            method='COBYLA',
            options={'maxiter': maxiter, 'disp': True}
        )
        self.weights = opt_res.x
        return self

# ==========================================
# 4. Training and Evaluation
# ==========================================
vqc = CirqVQC(circuit, qubits, x_symbols, theta_symbols)
vqc.fit(X_train, y_train, maxiter=30)

y_pred_train = vqc.predict(X_train)
y_pred_test = vqc.predict(X_test)

print("\n=== Model Performance ===")
print(f"Train Accuracy: {accuracy_score(y_train, y_pred_train) * 100:.2f}%")
print(f"Test Accuracy:  {accuracy_score(y_test, y_pred_test) * 100:.2f}%\n")
print(classification_report(y_test, y_pred_test, target_names=["Class 0", "Class 1"]))
