## Prerrequisitos

Para ejecutar este proyecto, necesitas tener instalado **Miniconda** o **Anaconda**. 

* **CUDA:** Este proyecto requiere soporte para GPU (CUDA 11.7 o superior). Hay que tener los drivers de NVIDIA actualizados.
* **Sistema:** Probado en entornos Ubuntu (WSL).


## Clonar el repositorio
```bash
cd /ruta/a/tu/directorio/de/trabajo
cd COMPUTER_VISION/T3-Action-Slot-Entrega/
```


Para asegurar la reproducibilidad, recomendamos crear un entorno limpio, el repositorio original tenia muchas dependencias desactualizadas, y algunas que chocaban entre si. Me encargue de dejarlo con lo minimo necesario

```bash
conda create -n action_slot_clean python=3.9 -y
conda activate action_slot python=3.7.12 -y
```

Nota: el archivo ``requirements.txt`` no es el mismo que el del repositorio de github, es una version con incompatibildiades quitadas.

```bash
conda install pip
pip install -r requirements.txt
```

Para exportar use
```bash
conda env export > environment.yml
pip freeze > requirements.txt
```


## Carpeta models
La carpeta models fue tomada de [HCIS-Lab/Action-slot](https://github.com/HCIS-Lab/Action-slot), con la finaldiad de 

## Scripts Desarrollados y Herramientas

Para facilitar el uso del modelo con videos propios y depurar la arquitectura, creé los siguientes scripts personalizados:


### 1. Pruebas y Depuración del Modelo (`inferencefix.py`)
Creé este script como un entorno de pruebas para validar la configuración y los pesos del modelo sin depender de la carga de videos reales. Sus funciones principales son:
* Instanciar la arquitectura `ACTION_SLOT` con los argumentos exactos requeridos (backbone `x3d`, clases `ego` y `actor` para TACO) solucionando errores de inicialización.
* Cargar los pesos preentrenados del modelo (`taco_action_slot_best_model.pth`).
* Pasar un tensor ficticio (dummy video) a través de la red para comprobar que las dimensiones de los tensores de salida (Ego, Actor y máscaras de atención) sean estructuralmente correctas.

### 2. Preprocesamiento de Video (`rawToClean.py`)
Este archivo fue creado para estandarizar los videos crudos (raw) antes de procesarlos. Utiliza la librería `ffmpeg` para realizar las siguientes tareas automáticamente:
* Recortar los videos temporalmente (definiendo un inicio y una duración).
* Redimensionar la resolución espacial a 1280x720 y ajustar el framerate a 30 FPS.
* Eliminar las pistas de audio y re-codificar el video en formato H.264 (`yuv420p`) para asegurar la compatibilidad con el pipeline de visión.


### 3. Inferencia y Generación de Resultados (`inferenceVisual.py`)
Este es el script principal (End-to-End) para ejecutar el modelo sobre videos reales y generar demostraciones visuales. El script se encarga de:
* Muestrear los frames del video de entrada para que coincidan con la longitud de secuencia (`seq_len=16`) que espera el modelo, calculando los FPS de salida para mantener la velocidad real de reproducción.
* Ejecutar la inferencia para obtener las probabilidades de las acciones y las máscaras de atención espacio-temporales.
* Renderizar un video de salida (`.mp4`) donde se superponen mapas de calor (heatmaps) sobre los actores detectados usando interpolación bilineal y *alpha blending*, además de imprimir en pantalla las clases activas y sus niveles de confianza.