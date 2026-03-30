"""Constantes compartilhadas do projeto."""

# Mapeamento de nomes de classes para nomes de exibição
DISPLAY_NAMES = {
    "beans package": "Feijao",
    "pasta package": "Macarrao",
    "rice package": "Arroz",
    "beans_package": "Feijao",
    "pasta_package": "Macarrao",
    "rice_package": "Arroz",
}

# Cores para cada classe (BGR format for OpenCV)
CLASS_COLORS = {
    "Feijao": (80, 180, 255),
    "Macarrao": (110, 245, 140),
    "Arroz": (255, 215, 120),
}

# Configurações padrão de câmera
DEFAULT_CAMERA_WIDTH = 1280
DEFAULT_CAMERA_HEIGHT = 720
DEFAULT_CAMERA_BUFFER_SIZE = 1

# Configurações padrão de detecção
DEFAULT_CONFIDENCE = 0.7
DEFAULT_LINE_Y_RATIO = 0.6
DEFAULT_MIN_LABEL_VOTES = 3
