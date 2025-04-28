import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import cg, LinearOperator
import scipy.io
import os
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

# Crear directorio para salida
OUTPUT_DIR = "output_images"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Cargar la matriz de medición y el sinograma desde Data164.mat
print("Cargando Data164.mat...")
file_path = r"C:\Users\felip\Downloads\Data164.mat"
data = scipy.io.loadmat(file_path)
A_original = data['A']  # Guardar la matriz A original

A = data['A'].copy()  # Matriz de medición (hacer una copia)
m = data['m']  # Sinograma
print(f"Dimensiones de A: {A.shape}")
print(f"Dimensiones de m: {m.shape}")

# Calcular tamaño de imagen
N = int(np.sqrt(A.shape[1]))
print(f"Tamaño de imagen: {N}x{N}")

# Parámetro de regularización
alpha = 10  # Mismo valor que en el código MATLAB

# Configurar para obtener 120 proyecciones (todas)
print("Preparando reconstrucción con 120 proyecciones...")
m_vec = m.flatten(order='F')  # Aplanar en orden de columna (como en MATLAB)

# Definir el operador lineal A'*A + alpha*I
def matvec_full(v):
    Av = A @ v
    return A.T @ Av + alpha * v

linear_op = LinearOperator((A.shape[1], A.shape[1]), matvec=matvec_full)

# Calcular el término del lado derecho b = A'*m
b = A.T @ m_vec

# Resolver el sistema usando gradiente conjugado
print("Calculando reconstrucción con 120 proyecciones...")
x, exitCode = cg(linear_op, b, maxiter=1000)

if exitCode:
    print(f"Advertencia: CG no convergió completamente. Código de salida: {exitCode}")
else:
    print("CG convergió exitosamente para 120 proyecciones.")

# Reconstrucción usando 20 proyecciones (1 de cada 6)
print("Preparando reconstrucción con 20 proyecciones...")
mm, nn = m.shape

# Seleccionar sólo 1 de cada 6 columnas del sinograma
m2 = m[:, ::6]

# Crear índices para seleccionar las filas correspondientes de A
# Este es un paso crítico para que coincida con la implementación en MATLAB
ind = []
for i in range(nn//6):
    start_idx = i * 6 * mm
    ind.extend(range(start_idx, start_idx + mm))

# Seleccionar las filas de A correspondientes a las proyecciones seleccionadas
A2 = A[ind, :]
print(f"Dimensiones de A2: {A2.shape}")
print(f"Dimensiones de m2: {m2.shape}")

# Definir el nuevo operador lineal para la versión reducida
def matvec_reduced(v):
    Av = A2 @ v
    return A2.T @ Av + alpha * v

linear_op2 = LinearOperator((A2.shape[1], A2.shape[1]), matvec=matvec_reduced)

# Aplanar m2 en orden de columna
m2_vec = m2.flatten(order='F')

# Calcular el lado derecho
b2 = A2.T @ m2_vec

# Resolver el sistema usando gradiente conjugado
print("Calculando reconstrucción con 20 proyecciones...")
x2, exitCode2 = cg(linear_op2, b2, maxiter=1000) 

if exitCode2:
    print(f"Advertencia: CG no convergió completamente para 20 proyecciones. Código de salida: {exitCode2}")
else:
    print("CG convergió exitosamente para 20 proyecciones.")

# Reformar las soluciones en imágenes
img_full = np.reshape(x, (N, N), order='F')  # Usar orden de columna como en MATLAB
img_reduced = np.reshape(x2, (N, N), order='F')

# Ajustar contraste para visualización mejorada
def adjust_contrast(image):
    # Normalizar a [0, 1]
    min_val = np.min(image)
    max_val = np.max(image)
    normalized = (image - min_val) / (max_val - min_val) if max_val > min_val else image
    return normalized

img_full_adj = adjust_contrast(img_full)
img_reduced_adj = adjust_contrast(img_reduced)

# Crear visualización similar a la de MATLAB
plt.figure(figsize=(10, 8))

# Sinograma completo (120 proyecciones)
plt.subplot(2, 2, 1)
plt.imshow(m, cmap='gray', aspect='auto')
plt.title('Sinogram, 120 projections')
plt.axis('off')

# Reconstrucción completa (120 proyecciones)
plt.subplot(2, 2, 2)
plt.imshow(img_full_adj, cmap='gray')
plt.title('Tikhonov reconstruction,\n120 projections')
plt.axis('off')

# Sinograma reducido (20 proyecciones)
plt.subplot(2, 2, 3)
plt.imshow(m2, cmap='gray', aspect='auto')
plt.title('Sinogram, 20 projections')
plt.axis('off')

# Reconstrucción reducida (20 proyecciones)
plt.subplot(2, 2, 4)
plt.imshow(img_reduced_adj, cmap='gray')
plt.title('Tikhonov reconstruction,\n20 projections')
plt.axis('off')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "matlab_like_reconstruction.png"), dpi=300)
print(f"Figura guardada en {os.path.join(OUTPUT_DIR, 'matlab_like_reconstruction.png')}")

# Guardar imágenes individuales en alta resolución
plt.figure(figsize=(8, 8))
plt.imshow(img_full_adj, cmap='gray')
plt.title('Tikhonov reconstruction, 120 projections')
plt.axis('off')
plt.savefig(os.path.join(OUTPUT_DIR, "reconstruccion_120_proyecciones_mejorada.png"), dpi=300)

plt.figure(figsize=(8, 8))
plt.imshow(img_reduced_adj, cmap='gray')
plt.title('Tikhonov reconstruction, 20 projections')
plt.axis('off')
plt.savefig(os.path.join(OUTPUT_DIR, "reconstruccion_20_proyecciones_mejorada.png"), dpi=300)

print(f"Reconstrucción completada. Imágenes guardadas en: {os.path.abspath(OUTPUT_DIR)}")