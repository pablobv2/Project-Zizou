import cv2
from ultralytics import YOLO

def detect_objects_on_frame(frame, model, conf_threshold=0.5):
    """
    Realiza la detección de objetos en un único frame de vídeo.

    Args:
        frame (np.ndarray): El frame de vídeo (imagen NumPy).
        model (YOLO): El modelo YOLO cargado.
        conf_threshold (float): Umbral de confianza para las detecciones.

    Returns:
        tuple: Una tupla conteniendo:
            - np.ndarray: El frame con las detecciones dibujadas.
            - list: Lista de resultados de detección crudos de YOLO para este frame.
                  (Puede ser útil para análisis posteriores).
    """
    # Realizar la detección
    # stream=True puede ser más eficiente para secuencias como vídeos
    results = model(frame, conf=conf_threshold, verbose=False, stream=True)

    processed_frame = frame.copy() # Evitar modificar el frame original directamente

    # Procesar los resultados para este frame
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy().astype(int)

        # Dibujar las bounding boxes en el frame copiado
        for box, conf, cls_id in zip(boxes, confidences, class_ids):
            x1, y1, x2, y2 = map(int, box)

            # Validar que el ID de clase existe en los nombres del modelo
            if cls_id < len(model.names):
                label = f"{model.names[cls_id]} {conf:.2f}"
                # Dibujar rectángulo y texto
                cv2.rectangle(processed_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(processed_frame, label, (x1, y1 - 10 if y1 > 10 else y1 + 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            else:
                print(f"Advertencia: ID de clase {cls_id} fuera de rango para model.names")


    return processed_frame, results # Devolver frame procesado y resultados crudos