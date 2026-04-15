import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


df = pd.read_csv("data/diamonds.csv")   ## Aquí simplemente cargamos los datos
resumen = df.describe(include=[np.number]).T    ## Hacemos la traspuesta para que la tabla sea más legible
resumen.to_csv("output/ej1_descriptivo.csv", index=True)    ## Lo pasamos a .csv con los nombres de las filas

print("Output the df.info(): \n")
df.info() ## Este comando nos da el número de NaNs, los dtypes, y el uso en memoria

## Con este dataframe podemos guardar casi toda la información de df.info() en el .csv
info_df = pd.DataFrame({
    "column": df.columns,
    "non_null_count": df.notna().sum().values,
    "n_nulls": df.isna().sum().values,
    "dtype": df.dtypes.astype(str).values
})

info_df.to_csv("output/ej1_descriptivo.csv", mode="a", header=False, index=False)

## Esto nos dice si hay valores nulos en las dimensiones
print("\nOutput de print((df[['x', 'y', 'z', 'price', 'carat', 'table']] == 0).sum()) \n")
print((df[["x", "y", "z", "price", "carat", "table"]] == 0).sum()) 

## Como tenemos varias filas con dimensiones nulas, las eliminamos.
df = df[(df[["x", "y", "z"]] != 0).all(axis=1)]

## Ahora vamos a ver si todos los datos categóricos son correctos.
print("\n Comprobación de las variables categóricas:\n")
print(df["cut"].unique())
print(df["clarity"].unique())
print(df["color"].unique())
## Hemos comprobado que las variables categóricas no tienen problemas

numeric_df = df.select_dtypes(include = "number")   ## Hacemos un dataframe solo con los valores
                                                    ## numéricos del dataframe original

medias = numeric_df.mean()
medianas = numeric_df.median()
modas = numeric_df.mode()
desviacion_tipica = numeric_df.std()
varianza = numeric_df.var()
minimo = numeric_df.min()
maximo = numeric_df.max()
cuartiles = numeric_df.quantile([0.25, 0.5, 0.75]).T
iqr_objetivo = numeric_df["price"].quantile([0.25,0.75]).T  ## Rango intercuartílico
skewness = numeric_df.skew()
kurtosis = numeric_df.kurtosis()

with open("output/ej1_metricas", "w") as f:
    f.write(f"Medias:\n {medias}\n")
    f.write(f"Medianas:\n {medianas}\n")
    f.write(f"Modas:\n {modas}\n")
    f.write(f"Desviacion tipica:\n {desviacion_tipica}\n")
    f.write(f"Minimo:\n {minimo}\n")
    f.write(f"Maximo:\n {maximo}\n")
    f.write(f"Cuartiles:\n {cuartiles}\n")
    f.write(f"Rango intercuartilico de la variable objetivo:\n {iqr_objetivo}\n")
    f.write(f"Skewness:\n {skewness}\n")
    f.write(f"Kurtosis:\n {kurtosis}\n")

numeric_df.hist(figsize = (12,8), bins = 20)    ## Escogemos bins = 20 porque parece interpretable
plt.tight_layout()                              ## para todas las variables
plt.savefig("output/ej1_histogramas.png", dpi=300, bbox_inches="tight")
plt.close()     ## Para que no aparezcan en pantalla los histogramas

columnas_categoricas = df.select_dtypes(include = ["str", "object", "category"]).columns   ## .columns devuelve
                                                                                    ## devuelve el nombre
                                                                                    ## de las columnas
                                                                                    ## para hacer el loop
for columna in columnas_categoricas:
    plt.figure(figsize = (8, 5))
    sns.boxplot(data = df, x = columna, y = "price")
    plt.title(f"Price by {columna}")
    plt.tight_layout()
    plt.savefig(f"output/ej1_boxplot_{columna}")
    plt.close()

## Como podemos ver en los boxplots estamos trabajando con un dataset con muchos outliers, 
## así que es mejor no usar el z-score. Además los datos no están normalmente distribuidos.
## Como la cantidad de outliers es tan grande es mejor no deshacernos de ellos o tratarlos
## de ninguna manera porque entonces estaríamos perdiendo mucha información.
## Probablemente lo que esté ocurriendo es que incluso si el corte es malo si el tamaño es grande
## puede seguir teniendo un valor muy elevado, por ejemplo.
## Lidiaremos con este fenómeno en el ejercicio dos.


## Ahora trabajaremos con las frecuencias de las variables categóricas.
for columna in columnas_categoricas:
    counts = df[columna].value_counts()
    plt.figure(figsize = (6,6))
    plt.pie(counts, labels = counts.index, autopct="%1.1f%%", startangle=90)
    plt.title(f"Pie chart of {columna}")
    plt.tight_layout()
    plt.savefig(f"output/ej1_categoricas_{columna}.png", dpi=300, bbox_inches="tight")
    plt.close()

corr_matrix = numeric_df.corr(method="pearson")

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Heatmap de las correlaciones de Pearson")
plt.tight_layout()
plt.savefig("output/ej1_heatmap_correlacion.png", dpi=300, bbox_inches="tight")
plt.close()