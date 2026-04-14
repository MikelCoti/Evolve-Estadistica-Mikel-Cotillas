"""
=============================================================================
PRÁCTICA FINAL — EJERCICIO 3
Regresión Lineal Múltiple implementada desde cero con NumPy
=============================================================================

DESCRIPCIÓN
-----------
En este ejercicio debes implementar la función `regresion_lineal_multiple`
que ajusta un modelo de regresión lineal múltiple utilizando la solución
analítica de Mínimos Cuadrados Ordinarios (OLS):

    β = (XᵀX)⁻¹ Xᵀy

La función debe ser capaz de:
  1. Añadir el término independiente (intercepto) automáticamente.
  2. Calcular los coeficientes β₀, β₁, ..., βₙ.
  3. Devolver las predicciones ŷ para un conjunto de datos nuevo.
  4. Calcular las métricas de evaluación: MAE, RMSE y R².

LIBRERÍAS PERMITIDAS
--------------------
  - numpy   (cálculos matriciales)
  - matplotlib (visualización, opcional)

NO está permitido usar sklearn para el ajuste del modelo en este ejercicio.

SALIDAS ESPERADAS (carpeta output/)
------------------------------------
  - output/ej3_coeficientes.txt   → Coeficientes del modelo ajustado
  - output/ej3_metricas.txt       → MAE, RMSE y R² sobre datos de test
  - output/ej3_predicciones.png   → Gráfico Real vs. Predicho

=============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import os

# Crear carpeta de salida si no existe
os.makedirs("output", exist_ok=True)


# =============================================================================
# FUNCIÓN PRINCIPAL — COMPLETA ESTA SECCIÓN
# =============================================================================

def regresion_lineal_multiple(X_train, y_train, X_test):
    """
    Ajusta un modelo de Regresión Lineal Múltiple usando OLS y devuelve
    las predicciones sobre el conjunto de test.

    La solución analítica es:
        β = (XᵀX)⁻¹ Xᵀy

    Parámetros
    ----------
    X_train : np.ndarray de forma (n_train, p)
        Matriz de features de entrenamiento. Cada fila es una observación
        y cada columna es una variable predictora.
    y_train : np.ndarray de forma (n_train,)
        Vector de valores objetivo de entrenamiento.
    X_test : np.ndarray de forma (n_test, p)
        Matriz de features sobre la que se quiere predecir.

    Retorna
    -------
    coefs : np.ndarray de forma (p+1,)
        Vector de coeficientes ajustados [β₀, β₁, ..., βₚ].
        β₀ es el intercepto (término independiente).
    y_pred : np.ndarray de forma (n_test,)
        Predicciones del modelo sobre X_test.

    Pistas
    ------
    - Usa np.hstack o np.column_stack para añadir la columna de unos (intercepto).
    - Usa np.linalg.inv o np.linalg.lstsq para resolver el sistema.
    - np.linalg.lstsq es numéricamente más estable que invertir directamente.
    """

    ## ¿Por qué estamos añadiendo una columna de 0s? Porque el coeficiente implícito del 
    ## intercepto es 1 en la ecuación que queremos resolver, es decir, la fórmula es
    ## y = b_0 * 1 + ..., donde ese 1 es el vector de unos, y luego el intercepto se calcula
    ## en el paso dos.
    n = len(y_train)
    intercepto = np.ones((n,1))
    X_train_b = np.hstack([intercepto, X_train])  # ← Reemplaza None con tu implementación


    X_train_b_transpose = X_train_b.T
    X_train_b_transpose_times_X_train_b = np.matmul(X_train_b_transpose, X_train_b)
    inverse_matrix = np.linalg.inv(X_train_b_transpose_times_X_train_b)
    total_matrix = np.matmul(inverse_matrix, X_train_b_transpose)
    coefs = np.matmul(total_matrix, y_train)  # ← Reemplaza None con tu implementación

    ## Este paso es necesario por la misma razón por la que añadimos un array de 1s
    ## en el primer paso, es el coeficiente del intercepto que vamos a predecir
    n_test = len(X_test[:,1])
    intercepto_test = np.ones((n_test,1))
    X_test_b = np.hstack([intercepto_test, X_test])


    y_pred = np.matmul(X_test_b, coefs)


    return coefs, y_pred


# =============================================================================
# FUNCIONES DE MÉTRICAS — COMPLETA ESTA SECCIÓN
# =============================================================================

def calcular_mae(y_real, y_pred):
    """
    Calcula el Mean Absolute Error (MAE).

        MAE = (1/n) * Σ |y_real - y_pred|

    Parámetros
    ----------
    y_real : np.ndarray — Valores reales
    y_pred : np.ndarray — Valores predichos

    Retorna
    -------
    float — Valor del MAE
    """
    coeficiente = 1/len(y_real)
    resta = y_pred - y_real
    valores_absolutos = np.abs(resta)
    mae = np.sum(valores_absolutos)*coeficiente
    return mae


def calcular_rmse(y_real, y_pred):
    """
    Calcula el Root Mean Squared Error (RMSE).

        RMSE = sqrt((1/n) * Σ (y_real - y_pred)²)

    Parámetros
    ----------
    y_real : np.ndarray — Valores reales
    y_pred : np.ndarray — Valores predichos

    Retorna
    -------
    float — Valor del RMSE
    """
  
    coeficiente = 1/len(y_real)
    y_dif = y_pred - y_real
    suma = 0
    for element in y_dif:
        suma += element**2
    rmse = np.sqrt(coeficiente*suma)
    return rmse

def calcular_r2(y_real, y_pred):
    """
    Calcula el coeficiente de determinación R².

        R² = 1 - SS_res / SS_tot
        SS_res = Σ (y_real - y_pred)²
        SS_tot = Σ (y_real - ȳ)²

    Parámetros
    ----------
    y_real : np.ndarray — Valores reales
    y_pred : np.ndarray — Valores predichos

    Retorna
    -------
    float — Valor del R² (entre -∞ y 1; cuanto más cercano a 1, mejor)
    """

    suma_res = 0
    suma_tot = 0
    y_res = y_real - y_pred
    mean_array = np.full(y_real.shape, np.mean(y_real))
    y_tot = y_real - mean_array
    for element_res in y_res:
        suma_res += element_res**2
    for element_tot in y_tot:
        suma_tot += element_tot**2
    
    r2 = 1 - suma_res/suma_tot
    return r2


# =============================================================================
# FUNCIÓN DE VISUALIZACIÓN — COMPLETA ESTA SECCIÓN (OPCIONAL)
# =============================================================================

def graficar_real_vs_predicho(y_real, y_pred, ruta_salida="output/ej3_predicciones.png"):
    """
    Genera un scatter plot de Valores Reales vs. Valores Predichos.

    Un modelo perfecto produciría todos los puntos sobre la diagonal y=x.
    La dispersión alrededor de esa línea representa el error del modelo.

    Parámetros
    ----------
    y_real      : np.ndarray — Valores reales del test set
    y_pred      : np.ndarray — Predicciones del modelo
    ruta_salida : str        — Ruta donde guardar la imagen
    """
    # TODO: Implementa la visualización
    # Pistas:
    #   - plt.scatter(y_real, y_pred, alpha=0.6)
    #   - Dibuja la línea de referencia perfecta: y = x
    #   - Añade etiquetas a los ejes y título
    #   - Guarda con plt.savefig(ruta_salida, dpi=150, bbox_inches='tight')
    
    x_min = min(min(y_real),min(y_pred))
    x_max = max(max(y_real), max(y_pred))
    x = np.linspace(x_min, x_max, 100)
    y = x
    plt.scatter(y_real, y_pred, alpha = 0.6)
    plt.plot(x, y, color = "red")
    plt.xlabel("Valores reales")
    plt.ylabel("Valores predecidos")
    plt.title("Scatterplot de valores")
    plt.tight_layout()
    plt.savefig(ruta_salida, dpi=150, bbox_inches='tight')
    plt.close()


# =============================================================================
# MAIN — NO MODIFIQUES ESTE BLOQUE (es la prueba de referencia del profesor)
# =============================================================================

if __name__ == "__main__":

    # -------------------------------------------------------------------------
    # Datos sintéticos con semilla fija para reproducibilidad
    # -------------------------------------------------------------------------
    SEMILLA = 42
    rng = np.random.default_rng(SEMILLA)

    n_muestras = 200
    n_features = 3

    # Generamos features aleatorias
    X = rng.standard_normal((n_muestras, n_features))

    # Coeficientes "reales" conocidos: β₀=5, β₁=2, β₂=-1, β₃=0.5
    coefs_reales = np.array([5.0, 2.0, -1.0, 0.5])

    # Variable objetivo con ruido gaussiano (σ=1.5)
    ruido = rng.normal(0, 1.5, n_muestras)
    y = coefs_reales[0] + X @ coefs_reales[1:] + ruido

    # -------------------------------------------------------------------------
    # Split Train / Test (80% / 20%) — sin mezclar aleatoriamente
    # -------------------------------------------------------------------------
    corte = int(0.8 * n_muestras)
    X_train, X_test = X[:corte], X[corte:]
    y_train, y_test = y[:corte], y[corte:]

    # -------------------------------------------------------------------------
    # Ajuste del modelo
    # -------------------------------------------------------------------------
    coefs, y_pred = regresion_lineal_multiple(X_train, y_train, X_test)

    # -------------------------------------------------------------------------
    # Métricas
    # -------------------------------------------------------------------------
    mae  = calcular_mae(y_test, y_pred)
    rmse = calcular_rmse(y_test, y_pred)
    r2   = calcular_r2(y_test, y_pred)

    # -------------------------------------------------------------------------
    # Mostrar resultados en consola
    # -------------------------------------------------------------------------
    print("=" * 50)
    print("RESULTADOS — Regresion Lineal Multiple (NumPy)")
    print("=" * 50)
    print(f"\nCoeficientes reales:   {coefs_reales}")
    print(f"Coeficientes ajustados: {coefs}")
    print(f"\nMetricas sobre test set:")
    print(f"  MAE  = {mae:.4f}")
    print(f"  RMSE = {rmse:.4f}")
    print(f"  R2   = {r2:.4f}")

    # -------------------------------------------------------------------------
    # RESULTADO DE REFERENCIA DEL PROFESOR
    # Con SEMILLA=42, los resultados esperados aproximados son:
    #   Coefs ajustados ≈ [5.0, 2.0, -1.0, 0.5]  (cercanos a los reales)
    #   MAE  ≈ 1.20  (±0.20 según implementación)
    #   RMSE ≈ 1.50  (±0.20 según implementación)
    #   R²   ≈ 0.80  (±0.05 según implementación)
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    # Guardar salidas
    # -------------------------------------------------------------------------

    # Fichero de coeficientes
    with open("output/ej3_coeficientes.txt", "w") as f:
        f.write("Regresion Lineal Multiple — Coeficientes ajustados\n")
        f.write("=" * 50 + "\n")
        nombres = ["Intercepto (b0)"] + [f"b{i+1} (feature {i+1})" for i in range(n_features)]
        for nombre, valor in zip(nombres, coefs):
            f.write(f"  {nombre}: {valor:.6f}\n")
        f.write("\nCoeficientes reales de referencia:\n")
        for nombre, valor in zip(nombres, coefs_reales):
            f.write(f"  {nombre}: {valor:.6f}\n")

    # Fichero de métricas
    with open("output/ej3_metricas.txt", "w") as f:
        f.write("Regresion Lineal Multiple — Metricas de evaluacion\n")
        f.write("=" * 50 + "\n")
        f.write(f"  MAE  : {mae:.6f}\n")
        f.write(f"  RMSE : {rmse:.6f}\n")
        f.write(f"  R2   : {r2:.6f}\n")

    # Gráfico
    graficar_real_vs_predicho(y_test, y_pred)

    print("\nSalidas guardadas en la carpeta output/")
    print("  → output/ej3_coeficientes.txt")
    print("  → output/ej3_metricas.txt")
    print("  → output/ej3_predicciones.png")
