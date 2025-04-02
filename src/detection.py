import cv2
import numpy as np
# No es necesario importar YOLO aquí si el modelo se pasa como argumento

# Opcional: Generar colores únicos para cada ID (si quieres diferenciar visualmente)
# Puedes usar una librería como matplotlib o crear una función simple
rng = np.random.default_rng(3) # Seed for reproducibility
colors = rng.uniform(0, 255, size=(100, 3)) # Pre-generate 100 random colors

def get_color(track_id):
    """Devuelve un color consistente para un ID de seguimiento dado."""
    idx = int(track_id) % len(colors)
    return tuple(map(int, colors[idx]))

def track_objects_on_frame(frame, model, conf_threshold=0.5, classes_to_track=None):
    """
    Realiza la detección y seguimiento de objetos en un único frame de vídeo
    utilizando el tracker especificado (ej. ByteTrack).

    Args:
        frame (np.ndarray): El frame de vídeo (imagen NumPy).
        model (YOLO): El modelo YOLO cargado.
        conf_threshold (float): Umbral de confianza para las detecciones iniciales.
        classes_to_track (list, optional): Lista de IDs de clase a seguir
                                           (ej. [0] para seguir solo la clase 0).
                                           Si es None, sigue todas las clases.

    Returns:
        tuple: Una tupla conteniendo:
            - np.ndarray: El frame con las detecciones y IDs de seguimiento dibujados.
            - list: Lista de resultados de tracking crudos de YOLO para este frame.
    """
    # Realizar detección y seguimiento
    # persist=True es ESENCIAL para que el tracker recuerde objetos entre frames
    # tracker='bytetrack.yaml' especifica el algoritmo de tracking
    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml", # o botsort.yaml
        conf=conf_threshold,
        classes=classes_to_track, # Filtra clases si se especifica
        verbose=False # Reduce la salida en consola durante el procesamiento
        )

    processed_frame = frame.copy()

    # Verificar si hay detecciones y IDs de seguimiento
    if results[0].boxes is not None and results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        confidences = results[0].boxes.conf.cpu().numpy()
        class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
        track_ids = results[0].boxes.id.int().cpu().numpy() # Obtener los IDs de seguimiento

        # Dibujar las bounding boxes y los IDs de seguimiento
        for box, conf, cls_id, track_id in zip(boxes, confidences, class_ids, track_ids):
            x1, y1, x2, y2 = map(int, box)

            # Obtener color para el ID (opcional, puedes usar un color fijo)
            track_color = get_color(track_id)
            # track_color = (0, 255, 0) # O usar siempre verde

            # Validar que el ID de clase existe en los nombres del modelo
            if cls_id < len(model.names):
                class_name = model.names[cls_id]
                label = f"ID {track_id}: {class_name} {conf:.2f}"

                # Dibujar rectángulo
                cv2.rectangle(processed_frame, (x1, y1), (x2, y2), track_color, 2)

                # Calcular tamaño del texto para el fondo
                (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                # Dibujar un rectángulo de fondo para el texto
                cv2.rectangle(processed_frame, (x1, y1 - text_height - baseline), (x1 + text_width, y1), track_color, -1)
                # Dibujar texto (blanco sobre fondo de color)
                cv2.putText(processed_frame, label, (x1, y1 - baseline),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            else:
                print(f"Advertencia: ID de clase {cls_id} fuera de rango para model.names")

    # results[0] contiene la información del primer (y único en este caso) frame procesado
    return processed_frame, results[0]