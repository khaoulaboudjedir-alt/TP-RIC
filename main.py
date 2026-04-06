import numpy as np
import matplotlib.pyplot as plt

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
ax.set_title("Scatter plot 3D — Vérité terrain")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")

plt.tight_layout()
plt.show()