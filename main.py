import cv2
from ultralytics import YOLO

def detect_players(image_path, model_path, conf_threshold=0.5):
    # Cargar el modelo personalizado
    model = YOLO(model_path)
    
    # Cargar la imagen
    frame = cv2.imread(image_path)
    
    # Realizar la detección
    results = model(frame, conf=conf_threshold)
    
    # Procesar los resultados
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy().astype(int)
        
        # Dibujar las bounding boxes
        for box, conf, cls_id in zip(boxes, confidences, class_ids):
            x1, y1, x2, y2 = map(int, box)
            
            # Dibujar rectángulo y texto
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{model.names[cls_id]} {conf:.2f}"
            cv2.putText(frame, label, (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
    
    # Guardar imagen con detecciones (sin mostrar ventanas)
    cv2.imwrite("result.jpg", frame)
    print("Detección completada. Imagen guardada como 'resultado.jpg'")

# Uso del código
detect_players(
    image_path="test.jpg",  # Ruta de tu imagen de prueba
    model_path="/home/pablo/MLFootball/trained_model.pt",         # Ruta de tu modelo entrenado
    conf_threshold=0.5            # Umbral de confianza
)