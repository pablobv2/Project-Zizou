# src/utils.py

import cv2

def draw_boxes(frame, boxes, confidences, class_ids, class_names):
    """
    Dibuja bounding boxes y etiquetas en un frame.
    (Esta lógica ahora está dentro de detect_objects_on_frame,
     pero podrías separarla aquí si se vuelve más compleja)
    """
    processed_frame = frame.copy()
    for box, conf, cls_id in zip(boxes, confidences, class_ids):
        x1, y1, x2, y2 = map(int, box)

        if cls_id < len(class_names):
             label = f"{class_names[cls_id]} {conf:.2f}"
             cv2.rectangle(processed_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
             cv2.putText(processed_frame, label, (x1, y1 - 10 if y1 > 10 else y1 + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        else:
             print(f"Advertencia: ID de clase {cls_id} fuera de rango para class_names")

    return processed_frame