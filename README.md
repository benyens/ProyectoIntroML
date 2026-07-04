# Clasificación Automática de Subgéneros de Jazz

## Descripción del Proyecto
Este repositorio contiene el pipeline y los experimentos para clasificar subgéneros de Jazz a partir del dataset FMA.

El enfoque actual compara dos familias de modelos para una tarea de clasificacion multiclase:
1. **Random Forest** con features tabulares (MFCC, delta, delta-delta, chroma, rolloff y ZCR), con balanceo de clases mediante **SMOTE**.
2. **CNN 2D** sobre espectrogramas Mel normalizados.

El notebook principal reporta metricas por clase, accuracy global, matrices de confusion y analisis de importancia de variables.

---

## Estructura del Repositorio

```text
ProyectoIntroML/
|
|-- data/                   # Datos, audios y matrices .npy
|-- notebooks/
|   |-- model.ipynb         # Notebook principal de entrenamiento y evaluacion
|-- src/
|   |-- download_data.py    # Descarga y filtrado inicial del dataset
|   |-- extract_features.py # Extraccion de features y generacion de matrices
|-- report/                 # Informe y figuras exportadas
|-- requirements.txt        # Dependencias de Python
|-- README.md               # Este archivo
```

---

## Requisitos y Entorno
Proyecto desarrollado en **Python 3.12**.

### 1. Crear y activar entorno virtual

Windows:
```cmd
python -m venv .venv
.venv\Scripts\activate
```

Linux/Mac:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Instalar dependencias
```cmd
pip install -r requirements.txt
```

---

## Obtencion de Datos (FMA)

El proyecto utiliza el dataset [Free Music Archive (FMA)](https://github.com/mdeff/fma). El script de descarga filtra y prepara los audios de interes para el pipeline.

```cmd
python src/download_data.py
```

---

## Ejecucion del Pipeline

### Paso 1: Extraccion de Features
Genera las matrices necesarias para entrenamiento y evaluacion.

```cmd
python src/extract_features.py
```

Esto produce, entre otros, los archivos:
- `data/X_tabular.npy`
- `data/X_cnn.npy`
- `data/y_encoded.npy`
- `data/classes.npy`

### Paso 2: Entrenamiento y Evaluacion
Abre y ejecuta el notebook principal:

1. Selecciona el kernel del entorno virtual.
2. Ejecuta `notebooks/model.ipynb` de arriba hacia abajo.

El notebook incluye:
- Balanceo de clases con SMOTE para el modelo tabular.
- Optimizacion de hiperparametros para Random Forest (RandomizedSearchCV).
- Entrenamiento de CNN y curvas de aprendizaje.
- Reportes de clasificacion y matrices de confusion.
- Analisis de importancia de features para Random Forest.

---

## Notas de Reproducibilidad
- El entrenamiento de CNN puede variar ligeramente entre ejecuciones (es recomendable subir la patience si entrega muy malos resultados).
- En Windows puede aparecer una advertencia de `joblib/loky` sobre conteo de nucleos fisicos; es una advertencia informativa y no invalida los resultados.

---

## Autor
- **Benjamin Renzo Ferrada Larach** - *202273061-7*
- Proyecto para la asignatura INF398.