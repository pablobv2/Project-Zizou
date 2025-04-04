import argparse
import os
from src.video_processor import process_video

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detecta y sigue jugadores en un vídeo usando YOLO y ByteTrack.")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Ruta al vídeo de entrada (.mp4, .avi, etc.)."
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Ruta para guardar el vídeo procesado (.mp4)."
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Ruta al archivo del modelo YOLO entrenado (.pt)."
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.5,
        help="Umbral de confianza para la detección inicial (ej: 0.5)."
    )
    parser.add_argument(
        "--classes",
        nargs='+', # Permite múltiples valores (ej: --classes 0 2)
        type=int,
        default=None, # Por defecto, sigue todas las clases
        help="IDs de las clases a seguir (ej: 0 para 'person' si es la clase 0). Si no se especifica, sigue todas las clases detectadas."
    )
    parser.add_argument(
        "--history",
        type=int,
        default=10, # Valor por defecto para la longitud del historial
        help="Number of past frames to consider for class stabilization (default: 10)."
    )

    args = parser.parse_args()

    # Validar existencia de archivos de entrada
    if not os.path.exists(args.input):
        print(f"Error: El archivo de vídeo de entrada no existe: {args.input}")
        exit(1)
    if not os.path.exists(args.model):
        print(f"Error: El archivo del modelo no existe: {args.model}")
        exit(1)

    # Ejecutar el procesamiento del vídeo con tracking
    process_video(
        input_video_path=args.input,
        output_video_path=args.output,
        model_path=args.model, # Usar la variable correcta
        conf_threshold=args.conf,
        classes_to_track=args.classes,
        history_len=args.history # Pasar el nuevo argumento
    )