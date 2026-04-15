## Fair, Good, Very Good, Premium, Ideal
## I1 (worst), SI2, SI1, VS2, VS1, VVS2, VVS1, IF (best)
## D (best) to J (worst)

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

## Leemos los datos y quitamos los 0s sin significado.
df = pd.read_csv("data/diamonds.csv")
df = df[(df[["x", "y", "z"]] != 0).all(axis=1)]

## Como estamos trabajando con variables ordinales tenemos que hacer un mapeo de estas a una
## escala numérica. La limitación que tiene esto es que estamos asumiendo que cada mejora en 
## escala es de la misma "cantidad", es decir, que las escalas son lineales.
mapa_calidad = {"Fair" : 1, "Good" : 2, "Very Good" : 3, "Premium" : 4, "Ideal" : 5}
mapa_claridad = {"I1" : 1, "SI2" : 2, "SI1" : 3, "VS2" : 4, "VS1" : 5, "VVS2" : 6, "VVS1" : 7, "IF" : 8}
mapa_color = {"J" : 1, "I" : 2, "H" : 3, "G" : 4, "F" : 5, "E" : 6, "D" : 7}

## Aquí estamos creando nuevas columnas.
df["calidad_encoded"] = df["cut"].map(mapa_calidad)
df["color_encoded"] = df["color"].map(mapa_color)
df["claridad_encoded"] = df["clarity"].map(mapa_claridad)

## Aquí estamos deshaciéndonos de las viejas columnas para que no corrompan nuestro proceso
## de regresión lineal
df_encoded = df.drop(columns = ["cut", "color", "clarity"])

## Aquí vamos a deshacernos de las variables muy correlacionadas entre sí para evitar la multicolinearidad.
df_encoded_dropped = df_encoded.drop(columns = ["y", "z", "carat"])

corr_matrix = df_encoded_dropped.select_dtypes(include = "number").corr(method="pearson")

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Heatmap de las correlaciones de Pearson")
plt.tight_layout()
plt.savefig("output/ej2_matriz_correlaciones")  ## Hemos hecho otra vez el heatmap para asegurarnos de que no hay más colinearidad
plt.close()

## Como los precios tienen escalas muy distintas a las demás variables vamos a 
## coger el logaritmo de los precios para mayor estabilidad numérica.

df_encoded_dropped["price"] = np.log(df_encoded_dropped["price"])

predictors = df_encoded_dropped.drop(columns = "price")
target = df_encoded_dropped["price"]

predictors_train, predictors_test, target_train, target_test = train_test_split(predictors, target, test_size = 0.2, random_state = 42)

modelo = LinearRegression()
modelo.fit(predictors_train, target_train)

target_predicted = modelo.predict(predictors_test)

mae = mean_absolute_error(target_test, target_predicted)
rmse = root_mean_squared_error(target_test, target_predicted)
r2 = r2_score(target_test, target_predicted)

with open("output/ej2_metricas_regresion.txt", "w", encoding="utf-8") as f:
    f.write(f"MAE: {mae}\n")
    f.write(f"RMSE: {rmse}\n")
    f.write(f"R^2: {r2}\n")

## Como podemos ver hay una no linearidad sospechosa a medida que los valores predichos aumentan.
## Digo que es sospechosa porque parece un fenómeno introducido por nosotros, quizás al 
## usar el logaritmo para escalar los datos. También es posible que nuestro modelo lineal
## no sea lo suficientemente efectivo cuando los precios de los diamantes son muy grandes
## y se requira de algún término polinomial.
residuos = target_test - target_predicted
plt.figure(figsize=(8, 5))
plt.scatter(target_predicted, residuos, alpha = 0.5)
plt.axhline(0, linestyle="--")
plt.xlabel("Valores predecidos")
plt.ylabel("Residuos")
plt.title("Gráfica de residuos")
plt.tight_layout()
plt.savefig("output/ej2_residuos", dpi=300, bbox_inches="tight")
plt.close()

## Ya que el carat determina en gran parte el precio de los diamantes, y como hemos 
## quitado el carat por su alta correlación con la variable x, decidimos introducir un
## término no lineal a nuestro modelo elevando el carat al cuadrado. 

## Como podemos ver en la gráfica y en las métricas, el resultado ha mejorado.

df_encoded_dropped["carat2"] = df_encoded["carat"]**2

predictors = df_encoded_dropped.drop(columns = "price")
target = df_encoded_dropped["price"]

predictors_train, predictors_test, target_train, target_test = train_test_split(predictors, target, test_size = 0.2, random_state = 42)

modelo = LinearRegression()
modelo.fit(predictors_train, target_train)

target_predicted = modelo.predict(predictors_test)

mae = mean_absolute_error(target_test, target_predicted)
rmse = root_mean_squared_error(target_test, target_predicted)
r2 = r2_score(target_test, target_predicted)

with open("output/ej2_metricas_regresion_carat2.txt", "w", encoding="utf-8") as f:
    f.write(f"MAE: {mae}\n")
    f.write(f"RMSE: {rmse}\n")
    f.write(f"R^2: {r2}\n")

residuos = target_test - target_predicted
plt.figure(figsize=(8, 5))
plt.scatter(target_predicted, residuos, alpha = 0.5)
plt.axhline(0, linestyle="--")
plt.xlabel("Valores predecidos")
plt.ylabel("Residuos")
plt.title("Gráfica de residuos con carat2")
plt.tight_layout()
plt.savefig("output/ej2_residuos_carat2", dpi=300, bbox_inches="tight")
plt.close()

## Ahora vamos a ver si los residuos siguen una distribución normal con respecto al 0.
## Comprobamos que los residuos no siguen una distribución normal.
## Pero también sabemos que el test de Jarque-Bera es muy sensible cuando hay muchos
## datos, que es nuestro caso, por lo que también dibujaremos un histograma.

from scipy.stats import jarque_bera

jb_stat, jb_pvalue = jarque_bera(residuos)

print("Jarque-Bera statistic:", jb_stat)
print("p-value:", jb_pvalue)

## Ahora dibujamos el histograma de los residuos. Visualmente podemos confirmar que efectivamente
## los residuos siguen una distribución normal.
from scipy.stats import norm

mu = residuos.mean()
sigma = residuos.std(ddof=1)

fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(residuos, bins=30, density=True, alpha=0.6)

x = np.linspace(residuos.min(), residuos.max(), 500)
ax.plot(x, norm.pdf(x, loc=mu, scale=sigma))

ax.axvline(0, linestyle="--")
ax.set_title("Histograma de residuos con normal ajustada")
ax.set_xlabel("Residuos")
ax.set_ylabel("Densidad")
plt.tight_layout()
fig.savefig("output/ej2_histograma_normalidad_residuos")
plt.close()