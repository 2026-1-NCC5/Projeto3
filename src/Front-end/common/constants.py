"""Constantes compartilhadas do projeto."""

# Mapeamento de nomes de classes para nomes de exibição
DISPLAY_NAMES = {
    "oleo": "oleo",
    "fuba": "fuba",
    "oil": "oleo",
    "cornmeal": "fuba",
    "oil package": "oleo",
    "cornmeal package": "fuba",
    "oil_package": "oleo",
    "cornmeal_package": "fuba",
}

# Cores para cada classe (BGR format for OpenCV)
CLASS_COLORS = {
    "oleo": (0, 215, 255),    # Amarelo/Dourado
    "fuba": (0, 140, 255),    # Alaranjado
}

# Configurações padrão de câmera
DEFAULT_CAMERA_WIDTH = 1280
DEFAULT_CAMERA_HEIGHT = 720
DEFAULT_CAMERA_BUFFER_SIZE = 1

# Configurações padrão de detecção
DEFAULT_CONFIDENCE = 0.7
DEFAULT_LINE_Y_RATIO = 0.6
DEFAULT_MIN_LABEL_VOTES = 3
