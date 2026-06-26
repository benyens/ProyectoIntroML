# Clasificación Automática de Subgéneros de Jazz mediante Deep Learning 

## Descripción del Proyecto
Este repositorio contiene el código fuente y los experimentos para la clasificación de *grano fino* de subgéneros de Jazz (Bebop, Swing, Fusion, etc.). El proyecto forma parte de la evaluación final de Inteligencia Artificial / Procesamiento de Audio.

El principal desafío técnico abordado es el **desbalanceo extremo de clases** y la **jerarquía multi-etiqueta** presente en datasets musicales del mundo real. 

Para resolverlo, implementamos un pipeline dual comparativo:
1. **Machine Learning Tradicional:** Un modelo `Random Forest` alimentado por características tabulares (MFCCs y Chroma) estabilizado mediante generación de datos sintéticos (**SMOTE**).
2. **Deep Learning:** Una Red Neuronal Convolucional (**CNN**) que procesa Espectrogramas de Mel en 2D, estabilizada mediante suavizado de pesos (*Weight Clipping*) y **Data Augmentation** (SpecAugment).

---

## Estructura del Repositorio

```text
ProyectroIntroML/
│
├── data/                   # Carpeta ignorada por Git. Aquí se alojarán los audios y matrices .npy.
├── notebooks/              
│   └── 01_exploracion_y_modelado.ipynb  # Cuaderno Jupyter interactivo con resultados finales.
├── src/                    
│   ├── download_data.py    # Script de ingesta y extracción quirúrgica del dataset.
│   └── extract_features.py # Pipeline de procesamiento de señal (audio a matrices).
├── report/                 # Código fuente LaTeX y figuras del informe final.
├── requirements.txt        # Dependencias de Python.
└── README.md               # Este archivo.
```

---

## Requisitos y Configuración del Entorno
Este proyecto fue desarrollado en **Python 3.12**. Se recomienda estrictamente el uso de un entorno virtual para evitar conflictos de dependencias (especialmente con TensorFlow).

### 1. Crear y activar el entorno virtual (Windows)
```cmd
python -m venv venv
venv\Scripts\activate
```
*(Para Linux/Mac use `python3 -m venv venv` y `source venv/bin/activate`)*

### 2. Instalar dependencias
```cmd
pip install -r requirements.txt
```

---

## Obtención de Datos (FMA Dataset)

Este proyecto utiliza el [Free Music Archive (FMA) Medium Dataset](https://github.com/mdeff/fma) (22 GB de audio original). Para reproducir los datos sin saturar su disco duro, hemos diseñado un script de extracción quirúrgica que descarga los metadatos y aísla *únicamente* los audios pertenecientes al género Jazz.

Para obtener los audios, ejecute:
```cmd
python src/download_data.py
```
*Nota: Este proceso requiere una conexión a internet estable y puede tardar varios minutos dependiendo de su ancho de banda.*

---

## Instrucciones de Ejecución

Una vez descargados los audios en la carpeta `data/audio_jazz/`, siga estos pasos para reproducir los experimentos:

### Paso 1: Extracción de Características (Feature Engineering)
Convierta los archivos `.mp3` crudos en matrices matemáticas (MFCCs y Espectrogramas de Mel). Este script guardará representaciones estáticas `.npy` en disco para acelerar el entrenamiento.
```cmd
python src/extract_features.py
```

### Paso 2: Modelado y Evaluación
Para visualizar la comparativa de modelos, el impacto de SMOTE y el Data Augmentation de la CNN, abra el cuaderno interactivo:
1. Asegúrese de que su entorno virtual esté seleccionado como el Kernel (núcleo) activo.
2. Ejecute el archivo `notebooks/model.ipynb`.
3. El cuaderno cargará las matrices generadas en el paso anterior de manera instantánea y mostrará los *Classification Reports* y Matrices de Confusión.

---

## Autor
* **Benjamín Renzo Ferrada Larach** - *202273061-7*
* Proyecto para la asignatura INF398.