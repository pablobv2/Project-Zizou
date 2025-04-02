import argparse
import os
from src.video_processor import process_video

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detecta jugadores en un vídeo usando YOLO.")
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
        help="Umbral de confianza para la detección (ej: 0.5)."
    )

    args = parser.parse_args()

    # Validar existencia de archivos de entrada
    if not os.path.exists(args.input):
        print(f"Error: El archivo de vídeo de entrada no existe: {args.input}")
        exit(1)
    if not os.path.exists(args.model):
        print(f"Error: El archivo del modelo no existe: {args.model}")
        exit(1)

    # Ejecutar el procesamiento del vídeo
    process_video(
        input_video_path=args.input,
        output_video_path=args.output,
        model_path=args.model,
        conf_threshold=args.conf
    )