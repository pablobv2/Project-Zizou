import cv2
import numpy as np

# Global dictionary para almacenar el filtro de Kalman de cada track
TRACK_KALMAN = {}

# --- Color Generation (Consistente - basado en track_id) ---
_rng = np.random.default_rng(42)
_colors = _rng.uniform(0, 255, size=(100, 3))

def get_color(track_id):
    idx = int(track_id) % len(_colors)
    return tuple(map(int, _colors[idx]))

def create_kalman_filter(initial_measurement):
    """
    Crea e inicializa un filtro de Kalman para rastrear la posición (x, y) con un modelo de velocidad constante.
    initial_measurement debe ser un np.array de forma [[x], [y]].
    """
    kalman = cv2.KalmanFilter(4, 2)
    # Matriz de transición del estado (modelo de velocidad constante)
    kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                        [0, 1, 0, 1],
                                        [0, 0, 1, 0],
                                        [0, 0, 0, 1]], dtype=np.float32)
    # Matriz de medición
    kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                         [0, 1, 0, 0]], dtype=np.float32)
    kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
    kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.5
    kalman.errorCovPost = np.eye(4, dtype=np.float32)
    kalman.statePost = np.array([[initial_measurement[0, 0]],
                                 [initial_measurement[1, 0]],
                                 [0],
                                 [0]], dtype=np.float32)
    return kalman

def track_objects_on_frame(frame, model, conf_threshold=0.5, classes_to_track=None, history_len=10):
    """
    Realiza detección y seguimiento con estabilización mediante un filtro de Kalman.
    Si en un frame no se detecta un objeto, se utiliza la predicción del filtro para dibujar la posición suavizada.
    
    Args:
        frame (np.ndarray): El frame del vídeo.
        model (YOLO): El modelo YOLO cargado.
        conf_threshold (float): Umbral de confianza para detecciones.
        classes_to_track (list, optional): Lista de IDs de clase a seguir.
        history_len (int): Número máximo de frames perdidos para continuar mostrando la detección.
    
    Returns:
        tuple: (processed_frame, results)
            - processed_frame (np.ndarray): Frame con las detecciones dibujadas.
            - results: Resultados crudos de YOLO para este frame.
    """
    results = model.track(
        frame, persist=True, tracker="botsort.yaml",
        conf=conf_threshold, classes=classes_to_track, verbose=False
    )
    processed_frame = frame.copy()
    current_detected_ids = set()
    
    if results[0].boxes is not None and results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        confidences = results[0].boxes.conf.cpu().numpy()
        current_class_ids = results[0].boxes.cls.cpu().numpy().astype(int)  # Clase en este frame

        # Unificar "goalkeeper" y "referee" en "player"
        if isinstance(model.names, dict):
            cls_name_to_id = {name: idx for idx, name in model.names.items()}
        else:
            cls_name_to_id = {name: idx for idx, name in enumerate(model.names)}

        if "player" in cls_name_to_id:
            player_id = cls_name_to_id["player"]
            goalkeeper_id = cls_name_to_id.get("goalkeeper")
            referee_id = cls_name_to_id.get("referee")
            current_class_ids = np.array([
                player_id if cls in (goalkeeper_id, referee_id) else cls
                for cls in current_class_ids
            ])
        
        track_ids = results[0].boxes.id.int().cpu().numpy()

        for box, conf, cls_id, track_id in zip(boxes, confidences, current_class_ids, track_ids):
            x1, y1, x2, y2 = map(int, box)
            # Usar la parte inferior central del bounding box como medición
            x_center = int((x1 + x2) / 2)
            y_bottom = int(y2)
            measurement = np.array([[np.float32(x_center)], [np.float32(y_bottom)]])
            
            # Inicializar o actualizar el filtro de Kalman para este track
            if track_id not in TRACK_KALMAN:
                kalman = create_kalman_filter(measurement)
                TRACK_KALMAN[track_id] = {
                    "kalman": kalman,
                    "missed": 0,
                    "bbox": (x1, y1, x2, y2),
                    "class": cls_id
                }
            else:
                kalman = TRACK_KALMAN[track_id]["kalman"]
                kalman.correct(measurement)  # Actualización con la medición
                TRACK_KALMAN[track_id]["missed"] = 0
                TRACK_KALMAN[track_id]["bbox"] = (x1, y1, x2, y2)
                TRACK_KALMAN[track_id]["class"] = cls_id
            
            # Predicción del filtro
            predicted = kalman.predict()
            pred_x, pred_y = int(predicted[0]), int(predicted[1])
            current_detected_ids.add(track_id)
            
            # Calcular dimensiones para la elipse basadas en el ancho del bbox
            width = x2 - x1
            ellipse_width = int(width * 0.5)
            ellipse_height = int(width * 0.15)
            light_blue = (200, 200, 255)
            
            # Dibujar la elipse en la posición predicha
            cv2.ellipse(
                processed_frame,
                center=(pred_x, pred_y),
                axes=(ellipse_width, ellipse_height),
                angle=0,
                startAngle=0,
                endAngle=360,
                color=light_blue,
                thickness=2
            )
            
    # Para los tracks no detectados en el frame actual, se predice su posición
    tracks_to_remove = []
    for track_id, info in TRACK_KALMAN.items():
        if track_id not in current_detected_ids:
            kalman = info["kalman"]
            predicted = kalman.predict()
            pred_x, pred_y = int(predicted[0]), int(predicted[1])
            info["missed"] += 1
            if info["missed"] <= history_len:
                x1, y1, x2, y2 = info["bbox"]
                width = x2 - x1
                ellipse_width = int(width * 0.5)
                ellipse_height = int(width * 0.15)
                light_blue = (200, 200, 255)
                cv2.ellipse(
                    processed_frame,
                    center=(pred_x, pred_y),
                    axes=(ellipse_width, ellipse_height),
                    angle=0,
                    startAngle=0,
                    endAngle=360,
                    color=light_blue,
                    thickness=2
                )
            else:
                tracks_to_remove.append(track_id)
    
    # Eliminar los tracks que han superado el límite de frames sin detección
    for track_id in tracks_to_remove:
        del TRACK_KALMAN[track_id]
    
    return processed_frame, results[0]
