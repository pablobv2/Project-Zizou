import cv2
import numpy as np

# --- Definiciones de Color y Equipo ---

# Define los colores BGR para dibujar las bounding boxes de cada equipo/categoría
# (Asegúrate de que estos colores sean claramente distinguibles)
TEAM_BOX_COLORS = {
    "Team A": (255, 0, 0),      # Azul para Team A
    "Team B": (0, 0, 255),      # Rojo para Team B
    "Goalkeeper": (0, 255, 255), # Amarillo para Portero (si lo detectas como clase separada o color)
    "Referee": (0, 0, 0),        # Negro para Árbitro (si lo detectas)
    "Unknown": (128, 128, 128)   # Gris para no asignados o ambiguos
}

# Define los rangos de color HSV aproximados para las camisetas de los equipos
# ¡¡ESTOS VALORES SON EJEMPLOS Y NECESITARÁS AJUSTARLOS!!**
# Usa herramientas como GIMP o selectores de color online para encontrar rangos HSV
# Formato: (Lower_H, Lower_S, Lower_V), (Upper_H, Upper_S, Upper_V)
TEAM_HSV_RANGES = {
    "Team A": ((95, 80, 50), (130, 255, 255)),   # Rango para azules/cianes fuertes
    "Team B": ((0, 100, 70), (10, 255, 255)),   # Rango para rojos (parte baja del H)
    # Añadir segunda parte del rojo si es necesario (parte alta del H)
    # "Team B_alt": ((170, 100, 70), (180, 255, 255)),
    "Goalkeeper": ((25, 80, 80), (35, 255, 255)), # Rango para amarillos/naranjas
    # Añade más equipos/categorías si es necesario
}

# --- Funciones de Procesamiento de Color ---

def get_dominant_color(image_roi, k=1):
    """
    Encuentra el color dominante en una ROI usando K-Means.

    Args:
        image_roi (np.ndarray): La región de interés de la imagen (en BGR).
        k (int): Número de clusters (colores dominantes a encontrar). Usar 1
                 suele ser suficiente para el color más dominante.

    Returns:
        tuple: El color dominante en formato BGR (int), o None si la ROI es inválida.
    """
    if image_roi is None or image_roi.shape[0] < 5 or image_roi.shape[1] < 5:
        # Evitar procesar ROIs muy pequeñas o inválidas
        return None

    try:
        # Convertir a formato adecuado para K-Means
        pixels = image_roi.reshape((-1, 3))
        pixels = np.float32(pixels)

        # Definir criterios y aplicar K-Means
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

        # 'centers' contiene los colores dominantes BGR
        # Convertir el color del centro principal a entero
        dominant_color_bgr = tuple(map(int, centers[0]))
        return dominant_color_bgr
    except cv2.error as e:
        # print(f"Error en K-Means (probablemente ROI muy pequeña o monocromática): {e}")
        # Si K-Means falla (a veces pasa con ROIs mínimas), podemos tomar el promedio como fallback
        if image_roi.size > 0:
            avg_color = np.mean(image_roi, axis=(0, 1))
            return tuple(map(int, avg_color))
        return None
    except Exception as e:
        print(f"Error inesperado en get_dominant_color: {e}")
        return None


def assign_team_by_hsv_color(bgr_color):
    """
    Asigna una etiqueta de equipo basada en un color BGR dominante,
    comparándolo con los rangos HSV predefinidos.

    Args:
        bgr_color (tuple): El color dominante en BGR.

    Returns:
        str: La etiqueta del equipo ("Team A", "Team B", "Unknown", etc.).
    """
    if bgr_color is None:
        return "Unknown"

    # Convertir el color BGR dominante a HSV
    hsv_color = cv2.cvtColor(np.uint8([[bgr_color]]), cv2.COLOR_BGR2HSV)[0][0]
    hue, saturation, value = hsv_color

    # Comprobar contra los rangos definidos
    assigned_team = "Unknown" # Por defecto
    for team, (lower, upper) in TEAM_HSV_RANGES.items():
        # Comprobación estándar de rango HSV
        lower_h, lower_s, lower_v = lower
        upper_h, upper_s, upper_v = upper

        # Manejo especial para el rojo (Hue puede estar cerca de 0 y 180)
        if team == "Team B": # Asumiendo que Team B es rojo
            # Comprobar si está en el rango bajo (0-10) O en el rango alto (170-180)
             in_range = (
                (hue >= lower_h and hue <= upper_h) or \
                (hue >= 170 and hue <= 180) # Ajusta este segundo rango si es necesario
             ) and \
             (saturation >= lower_s and saturation <= upper_s) and \
             (value >= lower_v and value <= upper_v)
        else:
            # Comprobación normal para otros colores
            in_range = (hue >= lower_h and hue <= upper_h) and \
                       (saturation >= lower_s and saturation <= upper_s) and \
                       (value >= lower_v and value <= upper_v)

        if in_range:
            assigned_team = team
            break # Asignar el primer equipo que coincida

    # Podrías añadir una comprobación adicional de saturación/valor mínimos
    # para evitar asignar equipos por colores muy deslavados o casi negros/blancos
    # if assigned_team != "Unknown" and (saturation < 50 or value < 40):
    #     return "Unknown" # Considerar color no fiable

    return assigned_team

# --- Función de Dibujo (opcional mover aquí, o mantener en detection.py) ---
# (La dejaremos en detection.py por ahora, pero usando los colores de aquí)

# --- Generador de Color para IDs (si todavía quieres diferenciar IDs únicos) ---
rng = np.random.default_rng(3) # Seed for reproducibility
_id_colors = rng.uniform(0, 255, size=(1000, 3)) # Pre-generate colors for track IDs

def get_id_color(track_id):
    """Devuelve un color consistente BGR para un ID de seguimiento dado."""
    idx = int(track_id) % len(_id_colors)
    return tuple(map(int, _id_colors[idx]))