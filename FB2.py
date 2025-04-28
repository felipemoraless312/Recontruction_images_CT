import numpy as np
import matplotlib.pyplot as plt
from skimage.data import shepp_logan_phantom
from scipy.interpolate import interp1d
from scipy.ndimage import rotate
from skimage.transform import rescale

# Generación del phantom
phantom = shepp_logan_phantom()
phantom_resized = rescale(phantom, scale=(512 / phantom.shape[0]), mode='reflect')

def radon_transform(image, angles):
    """
    Implementación manual de la Transformada Radon.
    """
    img_height, img_width = image.shape
    diagonal = int(np.ceil(np.sqrt(img_height**2 + img_width**2)))
    pad_width = (diagonal - img_width) // 2
    pad_height = (diagonal - img_height) // 2

    # Padding para ajustar a la diagonal
    padded_image = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width)), mode='constant')

    # Crear sinograma vacío
    sinogram = np.zeros((diagonal, len(angles)))

    for i, angle in enumerate(angles):
        # Rotar la imagen
        rotated_image = rotate(padded_image, angle, reshape=False, order=1)

        # Proyección (suma a lo largo de filas)
        projection = np.sum(rotated_image, axis=0)

        # Ajustar tamaño de la proyección al del sinograma
        if len(projection) < sinogram.shape[0]:
            # Rellenar con ceros si es más corta
            projection = np.pad(projection, (0, sinogram.shape[0] - len(projection)), mode='constant')
        else:
            # Recortar si es más larga
            projection = projection[:sinogram.shape[0]]

        sinogram[:, i] = projection

    return sinogram


def iradon_transform(sinogram, angles):
    """
    Implementación manual de la Transformada Inversa Radon.
    """
    diagonal, num_angles = sinogram.shape
    reconstructed = np.zeros((diagonal, diagonal))
    center = diagonal // 2

    for i, angle in enumerate(angles):
        # Reproyectar cada línea del sinograma
        line = sinogram[:, i]
        backprojected = np.tile(line, (diagonal, 1))
        backprojected = rotate(backprojected, -angle, reshape=False, order=1)

        # Sumar la proyección inversa
        reconstructed += backprojected

    # Normalización
    reconstructed /= len(angles)
    return reconstructed

# Configuración de ángulos
theta = np.linspace(0., 180., max(phantom_resized.shape), endpoint=False)

# Transformada Radon
parallel_proj = radon_transform(phantom_resized, theta)

# Parámetros para haz abanico
D = 256  # Distancia de la fuente al centro
def para2fan(parallel_proj, D, fan_coverage='cycle', fan_rotation_increment=None, 
             fan_sensor_geometry='arc', fan_sensor_spacing=None, interpolation='linear',
             parallel_sensor_spacing=1, parallel_coverage='halfcycle'):
    
    def form_gamma_vector(ploc, D, fan_sensor_spacing, fan_sensor_geometry):
        ploc_normalized = np.clip(ploc / D, -1, 1)
        if fan_sensor_geometry == 'line':
            floc = np.linspace(np.min(ploc), np.max(ploc), len(ploc))
            gamma_rad = np.arctan(floc / D)
        else:
            gamma_rad = np.arcsin(ploc_normalized)
        return gamma_rad

    def form_ptheta_vector(n, is_parallel_coverage_cycle):
        dptheta_deg = 360 / n if is_parallel_coverage_cycle else 180 / n
        ptheta = np.arange(n) * dptheta_deg
        return ptheta, dptheta_deg

    m, n = parallel_proj.shape
    ploc = np.linspace(-(m-1)/2, (m-1)/2, m) * parallel_sensor_spacing
    gamma_rad = form_gamma_vector(ploc, D, fan_sensor_spacing, fan_sensor_geometry)
    gamma_deg = np.degrees(gamma_rad)
    is_parallel_coverage_cycle = parallel_coverage == 'cycle'
    ptheta_deg, dptheta_deg = form_ptheta_vector(n, is_parallel_coverage_cycle)
    if fan_rotation_increment is None:
        fan_rotation_increment = dptheta_deg
    fan_proj = np.zeros((len(gamma_deg), n))
    t = D * np.sin(np.radians(gamma_deg))
    for i in range(n):
        interpolator = interp1d(ploc, parallel_proj[:, i], kind=interpolation, bounds_error=False, fill_value=0)
        fan_proj[:, i] = interpolator(t)
    
    return fan_proj, gamma_deg, ptheta_deg

def fan2para(fan_proj, gamma_deg, D, ploc):
    parallel_proj = np.zeros((len(ploc), fan_proj.shape[1])) 
    t = D * np.sin(np.radians(gamma_deg))
    for i in range(fan_proj.shape[1]):
        interpolator = interp1d(t, fan_proj[:, i], kind='linear', bounds_error=False, fill_value=0)
        parallel_proj[:, i] = interpolator(ploc)
    
    return parallel_proj

fan_proj, gamma_deg, fan_angles = para2fan(parallel_proj, D)

# Reconstrucción usando Transformada Inversa Radon
reconverted_parallel_proj = fan2para(fan_proj, gamma_deg, D, 
                                     np.linspace(-(parallel_proj.shape[0]-1)/2, 
                                                 (parallel_proj.shape[0]-1)/2, 
                                                 parallel_proj.shape[0]))
reconstructed_image = iradon_transform(reconverted_parallel_proj, theta)

# Visualización de resultados
plt.figure(figsize=(12, 6))

plt.subplot(1, 3, 1)
plt.imshow(parallel_proj, cmap='gray', extent=(0, 180, -phantom_resized.shape[0]//2, phantom_resized.shape[0]//2))
plt.title('Sinograma (Proyecciones Haz Paralelo)')
plt.xlabel('Ángulo de Rotación (grados)')
plt.ylabel('Posición del Sensor')

plt.subplot(1, 3, 2)
plt.imshow(fan_proj, cmap='gray', extent=(np.min(fan_angles), np.max(fan_angles), np.min(gamma_deg), np.max(gamma_deg)))
plt.title('Proyecciones Haz Abanico')
plt.xlabel('Ángulo de Rotación (grados)')
plt.ylabel('Posición del Sensor')

plt.subplot(1, 3, 3)
plt.imshow(reconstructed_image, cmap='gray')
plt.title('Imagen Reconstruida a partir de Haz Abanico')

plt.tight_layout()
plt.show()

plt.figure()
plt.imshow(phantom_resized, cmap='gray')
plt.title('Imagen Original')
plt.show()