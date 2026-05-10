"""Constantes compartilhadas do projeto."""

# Mapeamento de nomes de classes para nomes de exibição
DISPLAY_NAMES = {
    "oleo": "oleo",
    "fuba": "fuba",
    "oil": "oleo",
    "cornmeal": "fuba",
    "oil package": "oleo",
    "cornmeal package": "fuba",
    "cornmeal_package": "fuba",
    "rice package": "arroz",
    "beans package": "feijao",
    "pasta package": "macarrao",
    "rice": "arroz",
    "beans": "feijao",
    "pasta": "macarrao",
}

# Peso estimado por item detectado, usado no histórico pendente da câmera.
ESTIMATED_ITEM_WEIGHT_KG = {
    "oleo": 1.0,
    "fuba": 0.5,
    "arroz": 1.0, # Base (será ajustado dinamicamente)
    "feijao": 1.0,
    "macarrao": 0.5,
}

# Limiares de área (ratio) para detecção de peso (específico para arroz)
# Ratio = (Área do Box) / (Área da ROI)
RICE_WEIGHT_THRESHOLDS = {
    5.0: 0.18, # Acima de 18% da ROI é considerado 5kg
    2.0: 0.08, # Acima de 8% da ROI é considerado 2kg
    1.0: 0.00, # Caso contrário 1kg
}

# Cores para cada classe (BGR format for OpenCV)
CLASS_COLORS = {
    "oleo": (0, 191, 255),    # Dourado/Ouro
    "fuba": (0, 88, 234),     # Laranja (#ea580c em BGR)
    "arroz": (246, 130, 59),  # Azul (#3b82f6 em BGR)
    "feijao": (28, 28, 185),   # Vermelho (#b91c1c em BGR)
    "macarrao": (11, 158, 245), # Âmbar (#f59e0b em BGR)
}

# Configurações padrão de câmera
DEFAULT_CAMERA_WIDTH = 1280
DEFAULT_CAMERA_HEIGHT = 720
DEFAULT_CAMERA_BUFFER_SIZE = 1

# Configurações padrão de detecção
DEFAULT_CONFIDENCE = 0.7
DEFAULT_LINE_Y_RATIO = 0.6
DEFAULT_MIN_LABEL_VOTES = 3
