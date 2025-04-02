import cv2
import os
from ultralytics import YOLO
# Importar la función de TRACKING ahora
from src.detection import track_objects_on_frame
from tqdm import tqdm # Importar tqdm para la barra de progreso

def process_video(input_video_path, output_video_path, model_path, conf_threshold=0.5, classes_to_track=None):
    """
    Procesa un vídeo de entrada, detecta y sigue jugadores en cada frame
    y guarda un nuevo vídeo con las detecciones y IDs dibujados.

    Args:
        input_video_path (str): Ruta al vídeo de entrada.
        output_video_path (str): Ruta donde guardar el vídeo procesado.
        model_path (str): Ruta al modelo YOLO entrenado (.pt).
        conf_threshold (float): Umbral de confianza para las detecciones iniciales.
        classes_to_track (list, optional): Lista de IDs de clase a seguir.
                                           Si es None, sigue todas las clases.
    """
    # Cargar el modelo YOLO una sola vez
    try:
        model = YOLO(model_path)
        print(f"Modelo cargado exitosamente desde {model_path}")
        # Asegurarse de que model.names esté poblado (puede requerir una inferencia dummy a veces)
        if not hasattr(model, 'names') or not model.names:
             print("Intentando poblar model.names con una inferencia dummy...")
             _ = model(np.zeros((640, 640, 3), dtype=np.uint8)) # Inferencia dummy
        print(f"Clases detectadas por el modelo: {model.names}")

    except Exception as e:
        print(f"Error al cargar el modelo desde {model_path}: {e}")
        return

    # Abrir el vídeo de entrada
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"Error: No se pudo abrir el vídeo de entrada: {input_video_path}")
        return

    # Obtener propiedades del vídeo original
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0:
        print("Advertencia: No se pudo determinar el número total de frames. La barra de progreso podría no ser precisa.")
        # Estimar un número grande si es necesario para tqdm, o quitar la barra total
        # total_frames = None # Deshabilitar el total en tqdm

    print(f"Propiedades del vídeo de entrada:")
    print(f"  - Resolución: {frame_width}x{frame_height}")
    print(f"  - FPS: {fps:.2f}")
    if total_frames: print(f"  - Total Frames: {total_frames}")


    # Definir el codec y crear el objeto VideoWriter
    output_dir = os.path.dirname(output_video_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Directorio de salida creado: {output_dir}")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Codec para .mp4
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    if not out.isOpened():
        print(f"Error: No se pudo crear el archivo de vídeo de salida: {output_video_path}")
        cap.release()
        return

    print(f"Iniciando procesamiento y tracking con ByteTrack...")
    print(f"Salida se guardará en: {output_video_path}")

    # Usar tqdm para una barra de progreso
    # Usar 'with' asegura que la barra se cierre correctamente
    with tqdm(total=total_frames, desc="Procesando vídeo", unit="frame") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("\nFin del vídeo o error al leer frame.")
                break # Fin del vídeo o error

            # Procesar el frame actual para detectar y SEGUIR objetos
            # Pasamos el MISMO objeto 'model' en cada iteración
            processed_frame, tracking_results = track_objects_on_frame(
                frame, model, conf_threshold, classes_to_track
            )

            # Escribir el frame procesado en el vídeo de salida
            out.write(processed_frame)

            pbar.update(1) # Actualizar la barra de progreso

            # Opcional: Mostrar el vídeo mientras se procesa (ralentiza)
            # cv2.imshow('Video Tracking', processed_frame)
            # if cv2.waitKey(1) & 0xFF == ord('q'): # Presiona 'q' para salir
            #     print("\nProcesamiento interrumpido por el usuario.")
            #     break

    # Liberar recursos
    cap.release()
    out.release()
    print(f"\nProcesamiento completado. Vídeo guardado en: {output_video_path}")