import cv2
import os
from ultralytics import YOLO
from src.detection import detect_objects_on_frame # Importar la función de detección

def process_video(input_video_path, output_video_path, model_path, conf_threshold=0.5):
    """
    Procesa un vídeo de entrada, detecta jugadores en cada frame y guarda
    un nuevo vídeo con las detecciones dibujadas.

    Args:
        input_video_path (str): Ruta al vídeo de entrada.
        output_video_path (str): Ruta donde guardar el vídeo procesado.
        model_path (str): Ruta al modelo YOLO entrenado (.pt).
        conf_threshold (float): Umbral de confianza para las detecciones.
    """
    # Cargar el modelo YOLO una sola vez
    try:
        model = YOLO(model_path)
        print(f"Modelo cargado exitosamente desde {model_path}")
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

    print(f"Propiedades del vídeo de entrada:")
    print(f"  - Resolución: {frame_width}x{frame_height}")
    print(f"  - FPS: {fps:.2f}")
    print(f"  - Total Frames: {total_frames}")

    # Definir el codec y crear el objeto VideoWriter
    # Asegúrate de que el directorio de salida exista
    output_dir = os.path.dirname(output_video_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Directorio de salida creado: {output_dir}")

    # Codec común para MP4, puedes necesitar cambiarlo ('MJPG' para .avi)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    if not out.isOpened():
        print(f"Error: No se pudo crear el archivo de vídeo de salida: {output_video_path}")
        cap.release()
        return

    print(f"Procesando vídeo... Salida guardada en: {output_video_path}")
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Fin del vídeo o error al leer frame.")
            break # Fin del vídeo o error

        # Procesar el frame actual para detectar jugadores
        processed_frame, _ = detect_objects_on_frame(frame, model, conf_threshold)

        # Escribir el frame procesado en el vídeo de salida
        out.write(processed_frame)

        frame_count += 1
        # Opcional: Mostrar progreso cada N frames
        if frame_count % 100 == 0:
             print(f"Procesado frame {frame_count}/{total_frames}")

        # Opcional: Mostrar el vídeo mientras se procesa (puede ralentizar)
        # cv2.imshow('Video Processing', processed_frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'): # Presiona 'q' para salir
        #     print("Procesamiento interrumpido por el usuario.")
        #     break

    # Liberar recursos
    cap.release()
    out.release()
    cv2.destroyAllWindows() # Cierra ventanas si se usó cv2.imshow
    print(f"Procesamiento completado. Vídeo guardado en: {output_video_path}")