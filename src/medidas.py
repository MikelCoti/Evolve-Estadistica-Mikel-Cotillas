import numpy as np
import pandas as pd


def mean_evolve(data):
    suma = sum(data)
    media = suma / len(data)
    return media

def median_evolve(data):
    sorted_data = sorted(data)
    n = len(sorted_data)
    if n % 2 == 0:
        median = (sorted_data[n // 2 - 1] + sorted_data[n // 2]) / 2
    else:
        median = sorted_data[n // 2]
    return median
    
def variance_evolve(data):
    mean = mean_evolve(data)
    variance = sum((x - mean) ** 2 for x in data) / len(data)
    return variance

def std_dev_evolve(data):
    return variance_evolve(data) ** 0.5

def percentile_evolve(data, p):
    if not data:
        raise ValueError("data must not be empty")
    if not 0 <= p <= 100:
        raise ValueError("p must be between 0 and 100")

    x = sorted(data)
    n = len(x)

    if n == 1:
        return x[0]

    pos = (p / 100) * (n - 1)       ## Posición del percentil en la lista ordenada
    lower = int(pos)                ## Índice del valor inferior
    upper = min(lower + 1, n - 1)   ## Índice del valor superior
    weight = pos - lower            ## Peso para la interpolación

    return x[lower] * (1 - weight) + x[upper] * weight

def iqr_evolve(data):
    q1 = percentile_evolve(data, 25)
    q3 = percentile_evolve(data, 75)
    return q3 - q1  


if __name__ == "__main__":
    np.random.seed(42) 

    edad = list(np.random.randint(20, 60, 100))
    salario = list(np.random.normal(45000, 15000, 100))
    experiencia = list(np.random.randint(0, 30, 100))

    df = pd.DataFrame({'edad': edad,
                    'salario': salario,
                    'experiencia': experiencia})
    print(df.head())
    print(median_evolve(edad) == percentile_evolve(edad, 50))