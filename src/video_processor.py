# src/video_processor.py

import cv2
import os
from ultralytics import YOLO
from tqdm import tqdm
# Importar la función actualizada track_objects_on_frame con filtro de Kalman
from .detection import track_objects_on_frame

def process_video(input_video_path, output_video_path, model_path, conf_threshold=0.5, classes_to_track=None, history_len=10):
    """
    Procesa un vídeo, aplicando tracking y estabilización de clase mediante un filtro de Kalman.

    Args:
        input_video_path (str): Ruta al vídeo de entrada.
        output_video_path (str): Ruta donde guardar el vídeo procesado.
        model_path (str): Ruta al modelo YOLO.
        conf_threshold (float): Umbral de confianza.
        classes_to_track (list, optional): Clases a seguir.
        history_len (int): Número máximo de frames perdidos para seguir mostrando la detección.
    """
    # --- Cargar Modelo ---
    try:
        model = YOLO(model_path)
        print(f"Successfully loaded YOLO model from: {model_path}")
        if not hasattr(model, 'names') or not model.names:
             print("Populating model.names with a dummy inference...")
             _ = model(np.zeros((640, 640, 3), dtype=np.uint8))
        print(f"Model classes: {model.names}")
    except Exception as e:
        print(f"Error loading YOLO model: {e}")
        return

    # --- Abrir Vídeo ---
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"Error: Could not open input video: {input_video_path}")
        return

    # --- Propiedades y Writer ---
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
    if not out.isOpened():
        print(f"Error: Could not create output video writer for: {output_video_path}")
        cap.release()
        return

    print("-" * 30)
    print(f"Starting video processing with BoT-SORT and Kalman Filter stabilization (History={history_len})...")
    print(f"Output will be saved to: {output_video_path}")
    print("-" * 30)

    # --- Procesar Frames ---
    frame_count = 0
    try:
        with tqdm(total=total_frames, desc="Processing Frames", unit="frame", disable=total_frames is None) as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                processed_frame, _ = track_objects_on_frame(
                    frame, model,
                    conf_threshold, classes_to_track,
                    history_len=history_len  # Se pasa el parámetro para el filtro y gestión de historial
                )

                out.write(processed_frame)
                frame_count += 1
                pbar.update(1)

    except KeyboardInterrupt:
        print("\nProcessing interrupted by user (Ctrl+C).")
    finally:
        cap.release()
        out.release()
        print("-" * 30)
        print(f"Processing finished. {frame_count} frames processed.")
        print(f"Output video saved to: {output_video_path}")
