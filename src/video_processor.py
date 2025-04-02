import cv2
import os
from ultralytics import YOLO
# Asegúrate de que la importación refleje el cambio de nombre o funcionalidad
from src.detection import track_objects_on_frame
from tqdm import tqdm
import numpy as np # Necesario para la inferencia dummy si se usa

def process_video(input_video_path, output_video_path, model_path, conf_threshold=0.5, classes_to_track=None):
    """
    Procesa un vídeo: detecta, sigue, asigna equipos por color y guarda resultado.
    """
    # --- Carga del Modelo (sin cambios significativos) ---
    try:
        model = YOLO(model_path)
        print(f"Modelo cargado: {model_path}")
        # Force model.names population if needed
        if not hasattr(model, 'names') or not model.names:
             _ = model(np.zeros((640, 640, 3), dtype=np.uint8))
        print(f"Clases: {model.names}")
    except Exception as e:
        print(f"Error cargando modelo: {e}")
        return

    # --- Apertura de Vídeo y Configuración de Salida (sin cambios) ---
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"Error abriendo vídeo: {input_video_path}")
        return
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Info Vídeo: {frame_width}x{frame_height} @ {fps:.2f} FPS, {total_frames} frames")

    output_dir = os.path.dirname(output_video_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
    if not out.isOpened():
        print(f"Error creando vídeo de salida: {output_video_path}")
        cap.release()
        return

    # --- Procesamiento Frame a Frame ---
    print(f"Iniciando procesamiento con asignación de equipo por color...")
    # !!! Inicializar el diccionario de mapeo ID -> Equipo !!!
    track_team_mapping = {}

    with tqdm(total=total_frames if total_frames > 0 else None, desc="Procesando vídeo", unit="frame") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("\nFin del vídeo o error.")
                break

            # !!! Pasar y recibir track_team_mapping !!!
            processed_frame, _, track_team_mapping = track_objects_on_frame(
                frame,
                model,
                track_team_mapping, # Pasar el estado actual
                conf_threshold,
                classes_to_track
            )

            out.write(processed_frame)
            if total_frames > 0: pbar.update(1)

            # Opcional: Mostrar frame procesado
            # cv2.imshow('Team Color Tracking', processed_frame)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     print("\nInterrupción por usuario.")
            #     break

    # --- Liberar Recursos ---
    cap.release()
    out.release()
    print(f"\nProcesamiento completo. Vídeo guardado en: {output_video_path}")

    # Opcional: Imprimir el mapeo final ID-Equipo
    # print("\nMapeo final ID -> Equipo:")
    # for track_id, team in track_team_mapping.items():
    #     print(f"  ID {track_id}: {team}")