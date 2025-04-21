import shutil
import os
from huggingface_hub import hf_hub_download

repo_id = "pablobv2/DeepFootball"
filename = "trained_model.pt"
local_dir = "./models"

# Descarga en la caché (puedes especificar cache_dir=local_dir, pero aún así se mantendrá la estructura de caché interna)
file_path = hf_hub_download(repo_id=repo_id, filename=filename)

# Asegurar que el directorio destino exista
os.makedirs(local_dir, exist_ok=True)

# Copiar el archivo a la ubicación deseada
destination_path = os.path.join(local_dir, filename)
shutil.copy(file_path, destination_path)

print(f"Modelo copiado a: {destination_path}")
