# =====================================================
# TP-RIC — Cartographie d'une Fonction Mystère
# Réalisé par : KHAOULA BOUDJEDIR
# =====================================================
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
def out(filename):
    return os.path.join(OUTPUT_DIR, filename)
# =========================
# ÉTAPE 1 : GÉNÉRATION DU DATASET
# =========================
np.random.seed(42)
N = 2000
X_raw = np.random.uniform(-5, 5, (N, 2))  

print(f"Dataset généré : {N} points")
print(f"x ∈ [{X_raw[:,0].min():.2f}, {X_raw[:,0].max():.2f}]")
print(f"y ∈ [{X_raw[:,1].min():.2f}, {X_raw[:,1].max():.2f}]")
def f_target(x, y):
    return np.sin(np.sqrt(x**2 + y**2)) + 0.5 * np.cos(2*x + 2*y)

z_raw = f_target(X_raw[:, 0], X_raw[:, 1])  

print(f"\nAltitude z : min={z_raw.min():.4f}, max={z_raw.max():.4f}")
X_mean = X_raw.mean(axis=0)
X_std  = X_raw.std(axis=0)
X_norm = (X_raw - X_mean) / X_std
z_min = z_raw.min()
z_max = z_raw.max()
z_norm = (z_raw - z_min) / (z_max - z_min)
z_norm = z_norm.reshape(-1, 1)

print(f"Normalisation OK → X_norm shape={X_norm.shape}, z_norm shape={z_norm.shape}")
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

sc = ax.scatter(
    X_raw[:, 0],   
    X_raw[:, 1],   
    z_raw,         
    c=z_raw,      
    cmap='viridis',
    s=8,
    alpha=0.7
)

plt.colorbar(sc, ax=ax, shrink=0.5, pad=0.1, label="f(x,y)")
ax.set_title("Vérité Terrain — Scatter Plot 3D\nf(x,y) = sin(√(x²+y²)) + 0.5·cos(2x+2y)",
             fontsize=12, fontweight='bold')
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("f(x,y)")

plt.tight_layout()
plt.savefig(out("etape1_scatter3D.png"), dpi=120)
plt.show()

print("\n Étape 1 terminée !")
print("   Image sauvegardée : etape1_scatter3D.png")
# =========================
# ÉTAPE 2 : ARCHITECTURE DU RÉSEAU
# =========================
print("\n" + "="*55)
print("ÉTAPE 2 : Architecture du Réseau MLP")
print("="*55)

class MLP:
    """
    Perceptron Multicouche (MLP) flexible.
    Implémenté avec NumPy uniquement — aucun framework.

    Architecture recommandée : [2, 64, 64, 1]
      - 2  entrées  (x, y)
      - 64 neurones couche cachée 1
      - 64 neurones couche cachée 2
      - 1  sortie   (régression)

    Initialisation : He  → adapté à ReLU
    Activation     : ReLU sur les couches cachées
                     Linéaire (aucune) sur la couche de sortie
    Loss           : MSE (Erreur Quadratique Moyenne)
    """

    def __init__(self, layer_sizes):
        """
        Paramètre
        ---------
        layer_sizes : liste des tailles de chaque couche
                      ex. [2, 64, 64, 1]
        """
        self.layer_sizes = layer_sizes
        self.n_layers    = len(layer_sizes) - 1  
        self.W = []   
        self.b = []   

        for i in range(self.n_layers):
            fan_in = layer_sizes[i]
            fan_out = layer_sizes[i + 1]
            W_i = np.random.randn(fan_in, fan_out) * np.sqrt(2.0 / fan_in)
            
            b_i = np.zeros((1, fan_out))

            self.W.append(W_i)
            self.b.append(b_i)

            print(f"  Couche {i+1} : W{i+1} shape={W_i.shape}  "
                  f"b{i+1} shape={b_i.shape}  "
                  f"{'[ReLU]' if i < self.n_layers-1 else '[Linéaire]'}")

   
    def relu(self, z):
        return np.maximum(0, z)

    
    def relu_deriv(self, z):
        return (z > 0).astype(float)

    def forward(self, X):
        """
        Calcule la sortie du réseau pour une entrée X.

        Paramètre : X shape (batch, 2)
        Retourne  : y_pred shape (batch, 1)

        Cache Z (pré-activations) et A (activations)
        pour la backpropagation.
        """
        self.Z_cache = []    
        self.A_cache = [X]   

        A = X
        for i in range(self.n_layers):
            Z = A @ self.W[i] + self.b[i]      
            self.Z_cache.append(Z)

            if i < self.n_layers - 1:
               
                A = self.relu(Z)
            else:
                
                A = Z

            self.A_cache.append(A)

        return A  
    def mse_loss(self, y_pred, y_true):
       
        return np.mean((y_pred - y_true) ** 2)
    # =========================
    # ÉTAPE 3 : Rétropropagation (Backpropagation)
    # =========================
    def backward(self, y_true):
        """
        Calcule les gradients dW et db pour chaque couche.

        Paramètre : y_true shape (batch, 1)

        Règle de la chaîne appliquée de la sortie vers l'entrée :
          dL/dW[i] = A[i].T @ dZ[i]
          dL/db[i] = sum(dZ[i], axis=0)
          dL/dA[i] = dZ[i] @ W[i].T  (propagé à la couche précédente)
        """
        m = y_true.shape[0]  

        self.dW = [None] * self.n_layers
        self.db = [None] * self.n_layers
        dA = (2.0 / m) * (self.A_cache[-1] - y_true)   

        
        for i in reversed(range(self.n_layers)):

            Z      = self.Z_cache[i]    
            A_prev = self.A_cache[i]     

            if i < self.n_layers - 1:
                dZ = dA * self.relu_deriv(Z)       
            else:
                
                dZ = dA                            
            self.dW[i] = A_prev.T @ dZ             
            self.db[i] = dZ.sum(axis=0, keepdims=True)  
            dA = dZ @ self.W[i].T                  
    def update(self, lr):
       
        for i in range(self.n_layers):
            self.W[i] -= lr * self.dW[i]
            self.b[i] -= lr * self.db[i]
print("\nCréation du MLP [2, 64, 64, 1] :")
mlp = MLP(layer_sizes=[2, 64, 64, 1])
X_test = X_norm[:5]
y_pred_test = mlp.forward(X_test)

print(f"\nTest Forward Pass sur 5 exemples :")
print(f"  Entrée  shape : {X_test.shape}")
print(f"  Sortie  shape : {y_pred_test.shape}")
print(f"  Prédictions   : {y_pred_test.flatten().round(4)}")

y_true_test = z_norm[:5]
loss_avant = mlp.mse_loss(y_pred_test, y_true_test)
print(f"\nMSE (avant entraînement) : {loss_avant:.6f}")
print("  (valeur élevée normale — réseau pas encore entraîné)")

print("\n Étape 2 terminée !")
print("   • Initialisation He")
print("   • ReLU couches cachées / Linéaire sortie")
print("   • Loss MSE")
print("\n" + "="*55)
print("ÉTAPE 3 : Test de la Rétropropagation")
print("="*55)


y_pred_test = mlp.forward(X_test)
mlp.backward(y_true_test)

print("\nGradients calculés après un backward pass :")
for i in range(mlp.n_layers):
    print(f"  dW{i+1} shape={mlp.dW[i].shape}  "
          f"db{i+1} shape={mlp.db[i].shape}  "
          f"| |dW| = {np.linalg.norm(mlp.dW[i]):.4f}")

loss_avant = mlp.mse_loss(mlp.forward(X_test), y_true_test)
mlp.update(lr=0.005)
loss_apres = mlp.mse_loss(mlp.forward(X_test), y_true_test)

print(f"\nVérification mise à jour (lr=0.005) :")
print(f"  Loss avant update : {loss_avant:.6f}")
print(f"  Loss après update : {loss_apres:.6f}")
print(f"  → La loss {'a diminué ✅' if loss_apres < loss_avant else 'a augmenté ❌'}")

print("\n Étape 3 terminée !")
print("   • backward() : gradients calculés par règle de la chaîne (vectorisé)")
print("   • update()   : descente de gradient SGD")
# =========================
# ÉTAPE 4 : ENTRAÎNEMENT ET VISUALISATION
# =========================
print("\n" + "="*55)
print("ÉTAPE 4 : Entraînement et Visualisation")
print("="*55)
mlp = MLP(layer_sizes=[2, 64, 64, 1])

EPOCHS     = 1000
LR         = 0.005
BATCH_SIZE = 64
print(f"\nEntraînement : {EPOCHS} époques | lr={LR} | batch={BATCH_SIZE}")
print("-" * 45)

loss_history = []
N_train = X_norm.shape[0]

for epoch in range(EPOCHS):
    idx    = np.random.permutation(N_train)
    X_shuf = X_norm[idx]
    y_shuf = z_norm[idx]

    epoch_loss = 0.0
    n_batches  = 0
    for start in range(0, N_train, BATCH_SIZE):
        X_batch = X_shuf[start : start + BATCH_SIZE]   
        y_batch = y_shuf[start : start + BATCH_SIZE]   
        y_pred     = mlp.forward(X_batch)
        batch_loss = mlp.mse_loss(y_pred, y_batch)
        mlp.backward(y_batch)
        mlp.update(lr=LR)

        epoch_loss += batch_loss
        n_batches  += 1

    avg_loss = epoch_loss / n_batches
    loss_history.append(avg_loss)

    if (epoch + 1) % 100 == 0:
        print(f"  Époque {epoch+1:4d}/{EPOCHS}  |  Loss MSE = {avg_loss:.6f}")

print(f"\n Entraînement terminé — Loss finale : {loss_history[-1]:.6f}")
plt.figure(figsize=(9, 4))
plt.plot(loss_history, color='steelblue', linewidth=1.5)
plt.title("Courbe de la Loss (MSE) en fonction des époques", fontsize=13)
plt.xlabel("Époque")
plt.ylabel("MSE")
plt.grid(True, alpha=0.4)
plt.tight_layout()
plt.savefig(out("etape4_loss_curve.png"), dpi=120)
plt.show()
print("   Courbe de loss sauvegardée : etape4_loss_curve.png")


grid_n = 100
gx = np.linspace(-5, 5, grid_n)
gy = np.linspace(-5, 5, grid_n)
GX, GY = np.meshgrid(gx, gy)

grid_pts  = np.column_stack([GX.ravel(), GY.ravel()])   
grid_norm = (grid_pts - X_mean) / X_std                 


GZ_pred_norm = mlp.forward(grid_norm)                        
GZ_pred      = GZ_pred_norm * (z_max - z_min) + z_min       
GZ_pred      = GZ_pred.reshape(grid_n, grid_n)


GZ_true = f_target(GX, GY)


fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Test Final — Comparaison Fonction Originale vs MLP", fontsize=13, fontweight='bold')


c1 = axes[0].contourf(GX, GY, GZ_true, levels=60, cmap='viridis')
plt.colorbar(c1, ax=axes[0])
axes[0].set_title("Fonction Originale f(x,y)")
axes[0].set_xlabel("x"); axes[0].set_ylabel("y")


c2 = axes[1].contourf(GX, GY, GZ_pred, levels=60, cmap='viridis')
plt.colorbar(c2, ax=axes[1])
axes[1].set_title("Prédiction MLP [2, 64, 64, 1]")
axes[1].set_xlabel("x"); axes[1].set_ylabel("y")


error = np.abs(GZ_true - GZ_pred)
c3 = axes[2].contourf(GX, GY, error, levels=60, cmap='hot_r')
plt.colorbar(c3, ax=axes[2])
axes[2].set_title(f"Erreur Absolue  (MAE = {error.mean():.4f})")
axes[2].set_xlabel("x"); axes[2].set_ylabel("y")

plt.tight_layout()
plt.savefig(out("etape4_comparaison_heatmap.png"), dpi=120)
plt.show()
print("   Comparaison heatmap sauvegardée : etape4_comparaison_heatmap.png")

fig = plt.figure(figsize=(14, 5))
fig.suptitle("Vue 3D — Fonction Originale vs Prédiction MLP", fontsize=13, fontweight='bold')

ax1 = fig.add_subplot(121, projection='3d')
ax1.plot_surface(GX, GY, GZ_true, cmap='viridis', alpha=0.9)
ax1.set_title("Fonction Originale")
ax1.set_xlabel("x"); ax1.set_ylabel("y"); ax1.set_zlabel("f(x,y)")

ax2 = fig.add_subplot(122, projection='3d')
ax2.plot_surface(GX, GY, GZ_pred, cmap='plasma', alpha=0.9)
ax2.set_title("Prédiction MLP")
ax2.set_xlabel("x"); ax2.set_ylabel("y"); ax2.set_zlabel("f(x,y)")

plt.tight_layout()
plt.savefig(out("etape4_comparaison_3D.png"), dpi=120)
plt.show()
print("   Comparaison 3D sauvegardée : etape4_comparaison_3D.png")

print("\n Étape 4 terminée !")
print(f"   Loss finale  : {loss_history[-1]:.6f}")
print(f"   MAE sur grille : {error.mean():.4f}")
print("\n TP complet terminé avec succès !")