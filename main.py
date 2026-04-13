import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# ETAPE 1 : Generation du Dataset
# ============================================================
np.random.seed(42)
N = 2000

x = np.random.uniform(-5, 5, N)
y = np.random.uniform(-5, 5, N)

def f(x, y):
    return np.sin(np.sqrt(x**2 + y**2)) + 0.5 * np.cos(2*x + 2*y)

z = f(x, y)

x_mean, x_std = x.mean(), x.std()
y_mean, y_std = y.mean(), y.std()
z_mean, z_std = z.mean(), z.std()

x_norm = (x - x_mean) / x_std
y_norm = (y - y_mean) / y_std
z_norm = (z - z_mean) / z_std

X = np.column_stack([x_norm, y_norm])
Z = z_norm.reshape(-1, 1)

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter(x, y, z, c=z, cmap='viridis', s=2, alpha=0.6)
fig.colorbar(sc, ax=ax, shrink=0.5, label='z = f(x, y)')
ax.set_title("Scatter plot 3D - Verite terrain")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
plt.tight_layout()
plt.show()

# ============================================================
# ETAPE 2 : Architecture du Reseau
# ============================================================
class MLP:

    def __init__(self, layers=[2, 64, 64, 1]):
        self.layers = layers
        self.weights = []
        self.biases = []
        # Initialisation He
        for i in range(len(layers) - 1):
            W = np.random.randn(layers[i], layers[i+1]) * np.sqrt(2.0 / layers[i])
            b = np.zeros((1, layers[i+1]))
            self.weights.append(W)
            self.biases.append(b)

    def relu(self, z):
        return np.maximum(0, z)

    def relu_derivative(self, z):
        return (z > 0).astype(float)

    def forward(self, X):
        self.activations = [X]
        self.z_values = []
        A = X
        # Couches cachees - ReLU
        for i in range(len(self.weights) - 1):
            Z = A @ self.weights[i] + self.biases[i]
            self.z_values.append(Z)
            A = self.relu(Z)
            self.activations.append(A)
        # Couche de sortie - Lineaire
        Z = A @ self.weights[-1] + self.biases[-1]
        self.z_values.append(Z)
        self.activations.append(Z)
        return Z

    def mse_loss(self, y_pred, y_true):
        return np.mean((y_pred - y_true) ** 2)


# Creation du reseau
mlp = MLP([2, 64, 64, 1])

# Affichage
print("=" * 45)
print("      ARCHITECTURE DU RESEAU MLP")
print("=" * 45)
print(f"Couches          : {mlp.layers}")
print(f"Nombre de couches: {len(mlp.layers) - 1}")
print("-" * 45)
for i, (W, b) in enumerate(zip(mlp.weights, mlp.biases)):
    print(f"Couche {i+1} : W = {W.shape}  |  b = {b.shape}")
print("-" * 45)
y_pred = mlp.forward(X)
loss = mlp.mse_loss(y_pred, Z)
print(f"Output shape     : {y_pred.shape}")
print(f"Loss initiale    : {loss:.4f}")
print("=" * 45)