"""

 @autor: Felipe_Morales
HarmonicReconstructor BesselMethod
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import special
from scipy import interpolate
from skimage.data import shepp_logan_phantom
from skimage.transform import resize
from skimage.exposure import rescale_intensity
import time
import sys

def CRT2(f, p, phi, R, center):
    """
    Implementación de la Transformada de Radon Circular Directa

    Parámetros:
    f : array_like
        Imagen de entrada
    p : array_like
        Vector de radios
    phi : array_like
        Vector de ángulos
    R : float
        Radio máximo
    center : tuple
        Coordenadas del centro (x, y)

    Retorna:
    g : array_like
        Proyecciones (sinograma)
    """
    N = f.shape[0]
    N_p = len(p)
    N_phi = len(phi)
    N_gamma = 6*N
    g = np.zeros((N_p, N_phi))

    # Definición de ángulos para la integración sobre círculos
    gamma = np.linspace(0, 2*np.pi, N_gamma)
    p_cos_gamma = np.outer(np.cos(gamma), p)
    p_sin_gamma = np.outer(np.sin(gamma), p)

    # Interpolador para la imagen
    interp_func = interpolate.RectBivariateSpline(np.arange(N), np.arange(N), f)

    print("Calculando datos CRT...")

    for i in range(N_phi):
        sys.stdout.write(f"\rÁngulo: {i+1}/{N_phi}")
        sys.stdout.flush()

        # Cálculo de coordenadas para cada punto en círculos
        x = R * np.cos(phi[i]) + p_cos_gamma
        y = R * np.sin(phi[i]) + p_sin_gamma

        # Transformación al sistema de coordenadas de la imagen
        xi = y + center[1]
        yi = center[0] - x

        # Interpolación de valores de la imagen
        f_int = np.zeros_like(xi)

        for j in range(N_p):
            # Filtra coordenadas dentro de la imagen
            valid_indices = (yi[:, j] >= 0) & (yi[:, j] < N) & (xi[:, j] >= 0) & (xi[:, j] < N)
            if np.any(valid_indices):
                f_int[valid_indices, j] = interp_func(yi[valid_indices, j], xi[valid_indices, j], grid=False)

        # Integración sobre cada círculo
        g[:, i] = (gamma[1] - gamma[0]) * np.sum(f_int, axis=0)

    print("\nCálculo CRT finalizado.")

    # Ponderación por radio
    g = g * p[:, np.newaxis]

    return g

def besselj(n, x):
    """
    Wrapper para la función de Bessel de primera especie
    para manejar casos esapeciales similar a MATLAB.
    """
    return special.jv(n, x)

def besselderiv(x, m):
    """
    Calcula la derivada de la función de Bessel

    Parámetros:
    x : float o array
        Punto donde evaluar la derivada
    m : int
        Orden de la función de Bessel

    Retorna:
    derivvalue : float o array
        Valor de la derivada
    """
    # Usar directamente la fórmula de recurrencia para derivada de Bessel
    thederiv = (besselj(m-1, x) - besselj(m+1, x))/2

    # Asegurar que el resultado es real
    if np.any(np.imag(thederiv) != 0):
        print(f"Advertencia: derivada compleja para m={m}, x={x}")
        thederiv = np.real(thederiv)

    return thederiv

def BessDerivZerosBisect2(MM, KK, tol=1e-6):
    """
    Calcula los ceros de la primera derivada de la función de Bessel

    Parámetros:
    MM : array_like
        Vector de órdenes de la función de Bessel (≥ 0)
    KK : array_like
        Vector de índices de cero (≥ 1)
    tol : float, opcional
        Tolerancia para el valor de la derivada

    Retorna:
    jprimemk : array_like
        Raíces (ceros de la derivada)
    JofJ : array_like
        Valores de la función de Bessel en esos puntos
    """
    len_m = len(MM)
    len_k = len(KK)
    jprimemk = np.zeros((len_m, len_k))
    JofJ = np.zeros_like(jprimemk)

    # Tabla de ceros conocidos
    BesselDerivativeZerosT = np.array([
        [3.83170597020751, 7.01558666981561, 10.1734681350627, 13.3236919363142, 16.4706300508776],  # m=0
        [1.84118378134065, 5.33144277352503, 8.53631636634628, 11.7060049025920, 14.8635886339090],  # m=1
        [3.05423692822714, 6.70613319415845, 9.96946782308759, 13.1703708560161, 16.3475223183217],  # m=2
        [4.20118894121052, 8.01523659837595, 11.3459243107430, 14.5858482861670, 17.7887478660664],  # m=3
        [5.31755312608399, 9.28239628524161, 12.6819084426388, 15.9641070377315, 19.1960288000489],  # m=4
        [6.41561637570024, 10.5198608737723, 13.9871886301403, 17.3128424878846, 20.5755145213868],  # m=5
    ])

    row_bdz, col_bdz = BesselDerivativeZerosT.shape

    for m1 in range(len_m):
        m = MM[m1]
        for k1 in range(len_k):
            k = KK[k1]

            if (m < row_bdz) and (k-1 < col_bdz):
                # Usar tabla para aproximación inicial
                asymptroot = BesselDerivativeZerosT[int(m), int(k-1)]
                dx1 = 1e-9 if k <= 5 else 3e-5
                Aleft = asymptroot - dx1
                Aright = asymptroot + dx1
            else:
                if m == 0:
                    k = k + 1
                # Aproximación asintótica
                if k == 1:
                    oneterm = m + 0.8086165*m**(1/3) + 0.072490*m**(-1/3) - 0.05097*m**(-1) if m > 0 else 3.8317
                    oneterm = oneterm + 0.0094*m**(-5/3) if m > 0 else oneterm
                    asymptroot = oneterm
                else:
                    betapr = (m/2 + k - 3/4)*np.pi
                    mu = 4*m**2
                    asymptroot = betapr - (mu+3)/(8*betapr)
                    asymptroot = asymptroot - 4*(7*mu**2 + 82*mu - 9)/(3*(8*betapr)**3)

                # Bracket the root
                if k == 2:
                    Aleft = asymptroot - np.pi/(1.2 if abs(m) <= 1 else 1.0)
                elif k == 3:
                    Aleft = asymptroot - np.pi/2
                else:
                    Aleft = asymptroot - np.pi/(3 + 0.02*k)
                Aright = asymptroot + np.pi/3

            JprimeL = besselderiv(Aleft, m)
            JprimeR = besselderiv(Aright, m)

            # Asegurar que el intervalo contenga una raíz
            max_iter = 100
            iter_count = 0
            while JprimeL*JprimeR > 0 and iter_count < max_iter:
                iter_count += 1
                if JprimeL > 0:
                    Aleft = Aleft - (Aright - Aleft)/2
                    JprimeL = besselderiv(Aleft, m)
                else:
                    Aright = Aright + (Aright - Aleft)/2
                    JprimeR = besselderiv(Aright, m)

            # Método de bisección
            Amiddle = (Aleft + Aright)/2
            JprimeM = besselderiv(Amiddle, m)

            iter_count = 0
            while abs(JprimeM) > tol and iter_count < max_iter:
                iter_count += 1
                if JprimeL*JprimeM < 0:
                    Aright = Amiddle
                else:
                    Aleft = Amiddle
                    JprimeL = besselderiv(Aleft, m)
                Amiddle = (Aleft + Aright)/2
                JprimeM = besselderiv(Amiddle, m)

            jprimemk[m1, k1] = Amiddle
            JofJ[m1, k1] = JprimeM

    return jprimemk, JofJ

def optimized_gn_calculation(chc_g, p, z, L):
    """
    Cálculo optimizado de G_n usando muestreo reducido
    """
    print("Calculando G_n (versión optimizada)...")

    # Reducir el número de puntos z mediante muestreo
    sampling_factor_z = 4  # Calcular solo 1 de cada 4 puntos
    z_sampled = z[::sampling_factor_z]
    N_z_sampled = len(z_sampled)

    # Calcular G_n con puntos reducidos
    G_n_sampled = np.zeros((N_z_sampled, len(L)), dtype=complex)
    factor = (p[1]-p[0])/(2*np.pi)

    for i_z, z_val in enumerate(z_sampled):
        if i_z % 5 == 0:
            sys.stdout.write(f"\rProcesando z: {i_z+1}/{N_z_sampled}")
            sys.stdout.flush()

        bessel_vals = besselj(0, p * z_val)

        for i_l in range(len(L)):
            G_n_sampled[i_z, i_l] = factor * np.sum(chc_g[:, i_l] * bessel_vals)

    # Interpolar para obtener todos los puntos originales
    G_n = np.zeros((len(z), len(L)), dtype=complex)

    # Interpolar en la dimensión z
    for i_l in range(len(L)):
        # Interpolar parte real e imaginaria por separado
        real_interp = interpolate.interp1d(
            z_sampled, np.real(G_n_sampled[:, i_l]),
            bounds_error=False, fill_value="extrapolate", kind='cubic'
        )
        imag_interp = interpolate.interp1d(
            z_sampled, np.imag(G_n_sampled[:, i_l]),
            bounds_error=False, fill_value="extrapolate", kind='cubic'
        )

        G_n[:, i_l] = real_interp(z) + 1j * imag_interp(z)

    print("\nCálculo de G_n completado.")
    return G_n

def polar_to_cartesian_improved(f_pol_real_smooth, r, phi, N, center, original_image=None):
    """
    Mejora la reconstrucción de coordenadas polares a cartesianas
    para preservar mejor los bordes y orientación.

    Parámetros:
    - f_pol_real_smooth: Representación polar de la imagen
    - r: Vector de radios
    - phi: Vector de ángulos
    - N: Tamaño de la imagen de salida
    - center: Centro de la imagen
    - original_image: Imagen original para ajustar contraste (opcional)
    """
    # Crear imagen de salida
    f = np.zeros((N, N))

    # Ajustar el radio máximo para que coincida mejor con la imagen original
    # Utilizamos un factor basado en observaciones de las imágenes mostradas
    adjustment_factor = 2  # Ajustar según sea necesario
    max_r = adjustment_factor * N / 2

    # Escalar radios para ajustarse al tamaño de la imagen original
    r_scaled = r * (max_r / r[-1]) if r[-1] > 0 else r

    # Crear mallas para coordenadas cartesianas
    Y, X = np.meshgrid(np.arange(N), np.arange(N))

    # Cálculo de coordenadas polares para cada punto cartesiano
    dx = X - center[0]
    dy = Y - center[1]

    radius = np.sqrt(dx**2 + dy**2)
    theta = np.arctan2(dy, dx)

    # Normalizar ángulos a [0, 2π]
    theta = (theta + 2*np.pi) % (2*np.pi)

    # Usar interpolación bilineal para mejores resultados
    print("Aplicando interpolación mejorada...")

    # Para cada punto en el espacio cartesiano
    for i in range(N):
        if i % 20 == 0:  # Mostrar progreso cada 20 filas
            sys.stdout.write(f"\rProcesando fila {i}/{N}")
            sys.stdout.flush()

        for j in range(N):
            # Solo procesar puntos dentro del rango máximo
            if radius[i, j] <= r_scaled[-1]:
                # Interpolación bilineal
                # 1. Encontrar los índices más cercanos en r_scaled
                r_val = radius[i, j]
                r_idx_low = np.max(np.where(r_scaled <= r_val)[0]) if np.any(r_scaled <= r_val) else 0
                r_idx_high = r_idx_low + 1 if r_idx_low < len(r_scaled) - 1 else r_idx_low

                # 2. Encontrar los índices más cercanos en phi
                phi_val = theta[i, j]
                phi_idx_low = int((phi_val / (2*np.pi)) * len(phi))
                phi_idx_high = (phi_idx_low + 1) % len(phi)

                # 3. Verificar límites
                if (r_idx_low < f_pol_real_smooth.shape[0] and r_idx_high < f_pol_real_smooth.shape[0] and
                    phi_idx_low < f_pol_real_smooth.shape[1] and phi_idx_high < f_pol_real_smooth.shape[1]):

                    # 4. Calcular pesos para interpolación
                    dr = r_scaled[r_idx_high] - r_scaled[r_idx_low] if r_idx_low != r_idx_high else 1
                    r_weight = (r_val - r_scaled[r_idx_low]) / dr if r_idx_low != r_idx_high else 0
                    phi_weight = (phi_val - phi[phi_idx_low]) / (2*np.pi/len(phi)) if phi_idx_low != phi_idx_high else 0

                    # 5. Obtener valores para los cuatro puntos más cercanos
                    f00 = f_pol_real_smooth[r_idx_low, phi_idx_low]
                    f01 = f_pol_real_smooth[r_idx_low, phi_idx_high]
                    f10 = f_pol_real_smooth[r_idx_high, phi_idx_low]
                    f11 = f_pol_real_smooth[r_idx_high, phi_idx_high]

                    # 6. Aplicar interpolación bilineal
                    f[i, j] = (1-r_weight)*(1-phi_weight)*f00 + \
                              (1-r_weight)*phi_weight*f01 + \
                              r_weight*(1-phi_weight)*f10 + \
                              r_weight*phi_weight*f11

    print("\nInterpolación completada.")

    # Corregir la inversión vertical
    f = np.flipud(f)

    return f

def iCRT2(g, p, phi, R, N, center, original_image=None):
    """
    Implementación mejorada de la Inversión de la Transformada de Radon Circular

    Parámetros:
    g : array_like
        Proyecciones (sinograma)
    p : array_like
        Vector de radios
    phi : array_like
        Vector de ángulos
    R : float
        Radio máximo
    N : int
        Tamaño de la imagen reconstruida
    center : tuple
        Coordenadas del centro (x, y)
    original_image : array_like, opcional
        Imagen original para ajustar contraste

    Retorna:
    f : array_like
        Imagen reconstruida
    """
    N_p = len(p)
    N_phi = len(phi)
    N_z = 4*N_p
    Z = np.pi/(p[1]-p[0])

    # Pasaje a CHT (Transformada Circular Armónica)
    L = np.arange(-N_phi//2, N_phi//2)

    # Cálculo más preciso usando la expresión completa de la transformada de Fourier discreta
    chc_g = np.zeros((N_p, len(L)), dtype=complex)
    for l_idx, l in enumerate(L):
        chc_g[:, l_idx] = (phi[1]-phi[0])/(2*np.pi) * np.sum(g * np.exp(-1j * l * phi[np.newaxis, :]), axis=1)

    plt.figure(figsize=(10, 8))
    plt.imshow(np.abs(chc_g), aspect='auto', origin='lower')
    plt.colorbar()
    plt.title('Transformada Circular Armónica')
    plt.xlabel('Índice n')
    plt.ylabel('Índice p')
    plt.show()

    # Cálculo de los ceros de la derivada de Bessel
    r = np.linspace(0, R, N_phi)  # Asegurar que r tiene la misma longitud que N_phi
    N_k = 256  # Reducido para evitar problemas numéricos
    m = np.arange(0, np.floor(N_phi/2)+1)
    k = np.arange(1, N_k+1)
    print(f"Calculando ceros de la derivada de Bessel para m=[0...{len(m)-1}], k=[1...{len(k)}]")
    raiz, _ = BessDerivZerosBisect2(m, k)
    bessel_derivative_zeros = raiz
    zeros_bessel_derivative = (1/R) * bessel_derivative_zeros[:int(np.floor(N_phi/2))+1, :N_k]

    # Crear p_kn
    p_kn_neg = np.zeros((int(np.floor(N_phi/2)), N_k)) if len(m) > 1 else np.array([])
    if len(m) > 1:
        p_kn_neg = zeros_bessel_derivative[1:, :][::-1, :]
    p_kn_pos = zeros_bessel_derivative[:-1, :]
    p_kn = np.vstack((p_kn_neg, p_kn_pos)) if len(p_kn_neg) > 0 else p_kn_pos

    print(f"Forma p_kn: {p_kn.shape}")
    N_n = N_phi

    # Inversión en CHT
    z = (np.arange(1, N_z+1)/N_z) * Z
    N_z = len(z)

    # Cálculo de G_n usando método optimizado
    G_n = optimized_gn_calculation(chc_g, p, z, L)

    # Interpolación para g_n
    print("Calculando g_n...")
    g_n = np.zeros((N_k, len(L)), dtype=complex)

    # Para cada columna de G_n, crear un interpolador y aplicarlo a p_kn
    for i in range(len(L)):
        sys.stdout.write(f"\rProcesando L índice {i+1}/{len(L)}")
        sys.stdout.flush()

        if p_kn.shape[0] > i:  # Verificar que haya suficientes filas en p_kn
            # Crear un interpolador para cada columna de G_n
            interp = interpolate.interp1d(z, G_n[:, i], bounds_error=False, fill_value=0)

            # Aplicar el interpolador a los valores en p_kn[i, :]
            g_n[:, i] = interp(p_kn[i, :]) if i < p_kn.shape[0] else 0

    print("\nCálculo de g_n completado.")

    # Cálculo de coeficientes con manejo de errores numéricos
    print("Calculando coeficientes...")
    L2 = np.zeros((len(L), N_k))
    for i, l in enumerate(L):
        L2[i, :] = l

    # Calcular denominador con protección contra división por cero
    denominator = np.zeros((len(L), N_k), dtype=complex)
    for i, l in enumerate(L):
        for j in range(N_k):
            if i < p_kn.shape[0] and j < p_kn.shape[1]:
                p_val = p_kn[i, j]
                bessel_val = besselj(l, p_val*R)
                if abs(bessel_val) > 1e-10:
                    denominator[i, j] = (R**2 * p_val**2 - l**2) * bessel_val**3
                else:
                    denominator[i, j] = np.inf

    # Evitar divisiones por cero o valores muy pequeños
    denominator[np.abs(denominator) < 1e-10] = np.inf

    # Calcular coeficientes
    coef = np.zeros((len(L), N_k), dtype=complex)
    for i in range(len(L)):
        if i < p_kn.shape[0]:
            for j in range(N_k):
                if i < g_n.shape[0] and abs(denominator[i, j]) > 1e-10:
                    coef[i, j] = g_n[j, i] * (2 * p_kn[i, j]**2 / denominator[i, j])

    # Cálculo de funciones de Bessel para interpolación
    t = np.arange(0, np.max(p_kn)*R+0.5, 0.5)
    JBessel = np.zeros((len(L), len(t)))
    for i, l in enumerate(L):
        JBessel[i, :] = besselj(l, t)

    # Reconstrucción para cada armónico
    print("Realizando reconstrucción armónica...")
    chc_f = np.zeros((N_phi, N_phi), dtype=complex)

    for i in range(N_n):
        sys.stdout.write(f"\rArmónico: {i+1}/{N_n}")
        sys.stdout.flush()

        if i < p_kn.shape[0]:
            p2 = p_kn[i, :]
            valid_indices = p2 < Z
            p2 = p2[valid_indices]

            if len(p2) > 0:
                # Interpolación de funciones de Bessel
                jbessel = np.zeros((len(p2), len(r)))
                for j, p_val in enumerate(p2):
                    # Crear interpolador para la función de Bessel
                    if i < len(L):
                        interp = interpolate.interp1d(t, JBessel[i, :], bounds_error=False, fill_value=0)
                        jbessel[j, :] = interp(p_val * r)

                # Eliminar NaN
                jbessel = np.nan_to_num(jbessel)

                # Asegurarse de que el resultado de la multiplicación se ajuste a chc_f[:, i]
                if i < len(L) and i < coef.shape[0]:
                    valid_coef = coef[i, :len(p2)]
                    result = np.dot(valid_coef, jbessel)
                    # Asegurar que las dimensiones coincidan
                    if len(result) != N_phi:
                        # Redimensionar el resultado mediante interpolación
                        result_interp = np.interp(np.linspace(0, 1, N_phi),
                                              np.linspace(0, 1, len(result)),
                                              result)
                        chc_f[:, i] = result_interp
                    else:
                        chc_f[:, i] = result

    print("\nReconstrucción armónica completada.")

    """#Aplicar filtro gaussiano para suavizar
    chc_f_smooth = np.zeros_like(chc_f)
    chc_f_smooth.real = gaussian_filter(chc_f.real, sigma=1)
    chc_f_smooth.imag = gaussian_filter(chc_f.imag, sigma=1)"""
    chc_f_smooth = chc_f.copy()

    plt.figure(figsize=(10, 8))
    plt.imshow(np.abs(chc_f_smooth), aspect='auto', origin='lower')
    plt.colorbar()
    plt.title('Reconstrucción en Dominio Armónico')
    plt.xlabel('Índice n')
    plt.ylabel('Índice r')
    plt.show()

    # Retorno a coordenadas polares
    print("Calculando representación polar...")
    # Cálculo explícito de la transformada inversa
    f_pol = np.zeros((len(r), len(phi)), dtype=complex)
    for i_r in range(len(r)):
        for i_phi in range(len(phi)):
            for i_l, l in enumerate(L):
                if i_l < chc_f.shape[1]:
                    f_pol[i_r, i_phi] += chc_f[i_r, i_l] * np.exp(1j * l * phi[i_phi])

    # Aplicar restricción de valores negativos
    f_pol_real = np.real(f_pol)
    f_pol_real[f_pol_real < 0] = 0

    #f_pol_real_smooth = gaussian_filter(f_pol_real, sigma=1)
    # Sin suavizar
    f_pol_real_smooth = f_pol_real.copy()

    plt.figure(figsize=(10, 8))
    plt.imshow(f_pol_real_smooth, aspect='auto', extent=[0, 360, 0, r[-1]], origin='lower')
    plt.colorbar()
    plt.title('Representación Polar')
    plt.xlabel('Ángulo (grados)')
    plt.ylabel('Radio')
    plt.show()

    # Retorno a coordenadas cartesianas con método mejorado
    print("Transformando a coordenadas cartesianas...")

    # Usar función mejorada para transformación polar-cartesiana
    f = polar_to_cartesian_improved(f_pol_real_smooth, r, phi, N, center, original_image)

    return f

def main():
    """
    Función principal que ejecuta el proceso completo de reconstrucción tomográfica
    """
    # Parámetros
    N = 256
    R = N
    center = (N//2, N//2)

    # Vectores - Configuración optimizada
    N_p = 2*N
    N_phi = 180  # 180 proyecciones es estándar en tomografía
    p = np.linspace(1, 2*R-1, N_p)
    phi = np.linspace(0, 2*np.pi, N_phi, endpoint=False)

    print(f"Número de proyecciones angulares (N_phi): {N_phi}")
    print(f"Número de posiciones radiales (N_p): {N_p}")

    # Crear imagen de prueba (phantom de Shepp-Logan)
    try:
        image_test = shepp_logan_phantom(N)
    except TypeError:
        image_test = shepp_logan_phantom()
        image_test = resize(image_test, (N, N), anti_aliasing=True)

    # Visualización de la imagen original
    plt.figure(figsize=(10, 8))
    plt.imshow(image_test, cmap='gray')
    plt.colorbar()
    plt.title('Imagen Original')
    plt.xlabel('x (píxeles)')
    plt.ylabel('y (píxeles)')
    plt.show()

    # Transformada directa (CRT2)
    start_time = time.time()
    g = CRT2(image_test, p, phi, R, center)
    print(f"Tiempo de ejecución CRT2: {time.time() - start_time:.2f} segundos")

    # Visualización del sinograma
    plt.figure(figsize=(10, 8))
    plt.imshow(g, extent=[0, 360, p[-1], p[0]], aspect='auto', cmap='gray')
    plt.colorbar()
    plt.title('Proyecciones (Sinograma)')
    plt.xlabel('φ (grados)')
    plt.ylabel('p (píxeles)')
    plt.show()

    # Reconstrucción (iCRT2)
    start_time = time.time()
    f = iCRT2(g, p, phi, R, N, center, g)
    print(f"Tiempo de ejecución iCRT2: {time.time() - start_time:.2f} segundos")

    # Visualización de la imagen reconstruida
    plt.figure(figsize=(10, 8))
    plt.imshow(f, cmap='gray')
    plt.colorbar()
    plt.title('Imagen Reconstruida')
    plt.xlabel('x (píxeles)')
    plt.ylabel('y (píxeles)')
    plt.show()

    # Cálculo de errores
    mse = 100/N**2 * np.linalg.norm(image_test - f, 'fro')**2
    mae = 100/N**2 * np.linalg.norm(f - image_test, ord=1)
    NMSE = (100/N**2) * np.sum(np.abs(f - image_test)**2) / np.max(np.abs(image_test)**2)

    print(f"MSE: {mse:.6f} %")
    print(f"MAE: {mae:.6f} %")
    print(f"NMSE: {NMSE:.6f} %")

if __name__ == "__main__":
    main()
