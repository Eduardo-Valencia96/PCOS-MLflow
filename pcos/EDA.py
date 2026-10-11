import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def run_eda(df: pd.DataFrame, target: str = "PCOS (Y/N)") -> None:
    """Run exploratory data analysis on the dataset."""

    # First look at dataset
    display(df.head())

    print("\nDimensiones del dataset:")
    print(df.shape)

    print("\nNombres de las columnas:")
    print(df.columns.tolist())

    df.info()

    # Estadística descriptiva
    display(df.describe().T)

    # Valores faltantes
    missing = pd.DataFrame({
        "Missing Values": df.isnull().sum(),
        "Percentage": df.isnull().mean() * 100
    }).sort_values("Percentage", ascending=False)

    display(missing)

    # Variables numéricas y categóricas
    numeric_cols = df.select_dtypes(
        include=np.number
    ).columns.tolist()

    categorical_cols = df.select_dtypes(
        exclude=np.number
    ).columns.tolist()

    print("VARIABLES NUMÉRICAS:")
    print(numeric_cols)

    print("\nVARIABLES CATEGÓRICAS:")
    print(categorical_cols)

    # Valores únicos
    unique_values = pd.DataFrame({
        "Variable": df.columns,
        "Unique Values": [
            df[col].nunique(dropna=False)
            for col in df.columns
        ]
    })

    display(unique_values.sort_values("Unique Values"))

    # Distribución del target
    print("Distribución de PCOS:")
    display(df[target].value_counts())

    print("\nPorcentaje:")
    display(df[target].value_counts(normalize=True) * 100)

    plt.figure(figsize=(7, 5))
    sns.countplot(data=df, x=target)
    plt.title("Distribución de pacientes con PCOS")
    plt.xlabel("PCOS (0 = No, 1 = Sí)")
    plt.ylabel("Número de pacientes")
    plt.show()

    # Histogramas
    for col in numeric_cols:
        plt.figure(figsize=(7, 4))
        sns.histplot(data=df, x=col, kde=True)
        plt.title(f"Distribución de {col}")
        plt.xlabel(col)
        plt.ylabel("Frecuencia")
        plt.show()

    # Boxplots
    for col in numeric_cols:
        plt.figure(figsize=(8, 3))
        sns.boxplot(x=df[col])
        plt.title(f"Boxplot de {col}")
        plt.xlabel(col)
        plt.show()

    # Matriz de correlación
    correlation = df[numeric_cols].corr()

    plt.figure(figsize=(18, 14))
    sns.heatmap(
        correlation,
        cmap="coolwarm",
        center=0,
        annot=False
    )
    plt.title("Matriz de correlación")
    plt.show()

    # Correlación con PCOS
    target_corr = correlation[target].sort_values(
        ascending=False
    )

    display(target_corr)

    # Resumen del dataset
    summary = pd.DataFrame({
        "Tipo de dato": df.dtypes,
        "Valores únicos": df.nunique(),
        "Valores faltantes": df.isnull().sum(),
        "% faltantes": df.isnull().mean() * 100,
        "Duplicados": df.duplicated().sum()
    })

    display(summary)