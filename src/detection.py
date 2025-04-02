import cv2
import numpy as np
from src.utils import get_dominant_color, assign_team_by_hsv_color, TEAM_BOX_COLORS, get_id_color

def track_objects_on_frame(frame, model, track_team_mapping, conf_threshold=0.5, classes_to_track=None):
    """
    Detecta, sigue y asigna equipos a objetos en un frame, dibujando boxes coloreadas.

    Args:
        frame (np.ndarray): El frame de vídeo (imagen NumPy).
        model (YOLO): El modelo YOLO cargado.
        track_team_mapping (dict): Diccionario para almacenar la asignación equipo-ID.
                                   {track_id: team_label}. Se actualiza en esta función.
        conf_threshold (float): Umbral de confianza.
        classes_to_track (list, optional): IDs de clase a seguir.

    Returns:
        tuple: Una tupla conteniendo:
            - np.ndarray: El frame procesado.
            - list: Resultados crudos de YOLO para este frame.
            - dict: El diccionario track_team_mapping actualizado.
    """
    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        conf=conf_threshold,
        classes=classes_to_track,
        verbose=False
    )

    processed_frame = frame.copy()
    current_frame_results = results[0] # Resultados para este frame

    if current_frame_results.boxes is not None and current_frame_results.boxes.id is not None:
        boxes = current_frame_results.boxes.xyxy.cpu().numpy()
        confidences = current_frame_results.boxes.conf.cpu().numpy()
        class_ids = current_frame_results.boxes.cls.cpu().numpy().astype(int)
        track_ids = current_frame_results.boxes.id.int().cpu().numpy()

        for box, conf, cls_id, track_id in zip(boxes, confidences, class_ids, track_ids):
            x1, y1, x2, y2 = map(int, box)

            team_label = "Unknown"
            # 1. Comprobar si ya conocemos el equipo de este ID
            if track_id in track_team_mapping:
                team_label = track_team_mapping[track_id]
            else:
                # 2. Si es nuevo, intentar determinar el equipo por color
                # Definir la ROI de la camiseta (ajusta estos porcentajes según veas)
                roi_h = y2 - y1
                roi_w = x2 - x1
                # Tomar una ROI central superior (evitar cabeza y piernas)
                roi_y1 = y1 + int(roi_h * 0.15) # Empezar un poco abajo del borde superior
                roi_y2 = y1 + int(roi_h * 0.60) # Terminar antes de la mitad inferior
                roi_x1 = x1 + int(roi_w * 0.25) # Margen lateral
                roi_x2 = x2 - int(roi_w * 0.25) # Margen lateral

                # Asegurarse de que la ROI es válida
                roi_y1, roi_y2 = max(0, roi_y1), min(frame.shape[0], roi_y2)
                roi_x1, roi_x2 = max(0, roi_x1), min(frame.shape[1], roi_x2)

                if roi_y2 > roi_y1 and roi_x2 > roi_x1:
                    jersey_roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]

                    # Obtener color dominante y asignar equipo
                    dominant_bgr = get_dominant_color(jersey_roi)
                    if dominant_bgr:
                        team_label = assign_team_by_hsv_color(dominant_bgr)
                        # Guardar la asignación para futuras tramas
                        track_team_mapping[track_id] = team_label
                        # Opcional: Dibujar la ROI para depuración
                        # cv2.rectangle(processed_frame, (roi_x1, roi_y1), (roi_x2, roi_y2), (255, 255, 0), 1)
                else:
                     # Si la ROI no es válida, no podemos determinar color
                     track_team_mapping[track_id] = "Unknown" # Marcar como desconocido por ahora


            # 3. Obtener el color de la bounding box según el equipo asignado
            box_color = TEAM_BOX_COLORS.get(team_label, TEAM_BOX_COLORS["Unknown"])
            # Alternativa: Si quieres un color único por ID *además* del equipo:
            # id_color = get_id_color(track_id) # Podrías usarlo para el texto o un borde interior

            # 4. Dibujar
            if cls_id < len(model.names):
                class_name = model.names[cls_id]
                # Incluir etiqueta de equipo en el texto
                label = f"ID {track_id} ({team_label}): {class_name} {conf:.2f}"

                # Dibujar rectángulo con el color del equipo
                cv2.rectangle(processed_frame, (x1, y1), (x2, y2), box_color, 2)

                # Texto con fondo
                (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(processed_frame, (x1, y1 - text_height - baseline), (x1 + text_width, y1), box_color, -1)
                cv2.putText(processed_frame, label, (x1, y1 - baseline + 1),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1) # Texto blanco

            else:
                print(f"Advertencia: ID de clase {cls_id} fuera de rango")

    return processed_frame, current_frame_results, track_team_mapping