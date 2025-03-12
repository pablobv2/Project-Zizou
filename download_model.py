from huggingface_hub import hf_hub_download

# Configura estos parámetros
repo_id = "pablobv2/MLFootball"  # Reemplaza con tu usuario y nombre del repositorio
filename = "trained_model.pt"  # Nombre del archivo .pt que deseas descargar
local_dir = "./modelo_descargado"  # Directorio local para guardar el archivo
# token = "tu_token_hf"  # Opcional, solo si el modelo es privado

# Descargar el archivo .pt
file_path = hf_hub_download(
    repo_id=repo_id,
    filename=filename,
    cache_dir=local_dir,
#    token=token,
)

print(f"Modelo descargado en: {file_path}")