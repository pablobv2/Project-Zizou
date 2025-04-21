# Football Player Detection and Tracking

Un sistema de detección y seguimiento de jugadores en tiempo real para vídeos de fútbol basado en YOLOv11 y BoT-SORT.

## 📋 Descripción

Este proyecto implementa un sistema avanzado de visión por computadora para detectar y realizar seguimiento de jugadores, porteros y árbitros en vídeos de partidos de fútbol. Utiliza un modelo YOLOv11 personalizado, entrenado específicamente para el dominio del fútbol, junto con algoritmos de seguimiento de objetos y filtros de Kalman para proporcionar detecciones estables.

### Características principales:

- **Detección especializada**: Modelo YOLOv11 fine-tuned para identificar jugadores, porteros, árbitros y balones
- **Seguimiento robusto**: Implementación de BoTSORT derivado de ByteTrack para asociación de objetos entre frames
- **Estabilización**: Filtros de Kalman para suavizar trayectorias y manejar oclusiones

## 🚀 Instalación

### Requisitos

- Python 3.12
- PyTorch 2.6
- OpenCV 4.11
- Ultralytics (YOLOv11)

### Configuración del entorno

```bash
# Clonar el repositorio
git clone https://github.com/pablobv2/DeepFootball
cd DeepFootball

# Crear y activar entorno virtual (opcional pero recomendado)
python -m venv venv
source venv/bin/activate   # En Windows: venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt
```

### Descargar modelo pre-entrenado

Descarga nuestro modelo YOLOv11 fine-tuned para detección de fútbol:

```bash
mkdir -p models
# Descarga automática del modelo
python src/download_model.py
```

O descarga manualmente el modelo desde este enlace y colócalo en la carpeta `models/`.

## 💻 Uso

### Línea de comandos

```bash
python main.py --input videos/match.mp4 --output results/processed_match.mp4 --model models/trained_model.pt --conf 0.5
```

### Parámetros

- `--input`: Ruta al vídeo de entrada
- `--output`: Ruta donde guardar el vídeo procesado
- `--model`: Ruta al archivo del modelo YOLO entrenado (.pt)
- `--conf`: Umbral de confianza para la detección (default: 0.5)
- `--classes`: IDs de las clases a seguir (opcional, por defecto todas)
- `--history`: Número de frames para considerar en la estabilización (default: 10)


## 🧪 Resultados

El modelo ha sido entrenado con un dataset de imágenes de fútbol profesional y ha conseguido los siguientes resultados:

| Clase    | Precisión | Recall | mAP50 | mAP50-95 |
|----------|-----------|--------|-------|----------|
| General  | 0.849     | 0.802  | 0.832 | 0.603    |
| Balón    | 0.748     | 0.359  | 0.451 | 0.218    |
| Portero  | 0.902     | 0.962  | 0.957 | 0.718    |
| Jugador  | 0.943     | 0.971  | 0.987 | 0.798    |
| Árbitro  | 0.805     | 0.918  | 0.933 | 0.679    |

## 🔧 Estructura del proyecto

```
football-player-detection/
├── main.py                  # Punto de entrada principal
├── src/
│   ├── detection.py         # Funciones de detección con YOLO
│   ├── utils.py             # Funciones de utilidad
│   └── video_processor.py   # Procesamiento de vídeo
├── models/                  # Modelos pre-entrenados
├── assets/                  # Imágenes y recursos para documentación
└── requirements.txt         # Dependencias del proyecto
```

## 📊 Posibles aplicaciones futuras

- Análisis táctico de partidos
- Seguimiento de jugadores individuales
- Estadísticas automáticas de movimiento
- Mejora de retransmisiones deportivas
- Datasets etiquetados para análisis deportivo

## 📜 Licencia

Este proyecto está bajo la licencia MIT - ver el archivo LICENSE para más detalles.

## 🙏 Agradecimientos

- Ultralytics por el framework YOLO
- BoT-SORT por el algoritmo de seguimiento
- Roboflow por las herramientas de gestión de datasets

## 📧 Contacto

Si tienes preguntas o sugerencias, no dudes en abrir un issue o contactarme en:

- Pablo Barrio Val
- 📫 Email: pablo.barrio.val@gmail.com
- 🔗 LinkedIn: Pablo Barrio Val
