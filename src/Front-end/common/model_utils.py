"""Utilitários para resolução e carregamento de modelos."""

from pathlib import Path


def resolve_best_model(custom_path: str | None = None) -> Path:
    """
    Resolve o caminho para o melhor modelo treinado.
    
    Args:
        custom_path: Caminho opcional para um modelo específico.
                    Se None, procura pelo modelo padrão em detector/runs/modelo/weights/best.pt
    
    Returns:
        Path: Caminho absoluto para o arquivo best.pt
        
    Raises:
        FileNotFoundError: Se o modelo não for encontrado
    """
    root = Path(__file__).resolve().parent.parent

    # Se caminho customizado foi fornecido
    if custom_path:
        custom = Path(custom_path)
        if not custom.is_absolute():
            custom = root / custom
        if custom.exists():
            return custom
        raise FileNotFoundError(f"Modelo nao encontrado: {custom}")

    # Modelo padrão
    model_path = root / "detector" / "runs" / "modelo" / "weights" / "best.pt"
    
    if model_path.exists():
        return model_path
    
    raise FileNotFoundError(
        f"Modelo treinado nao encontrado em: {model_path}"
    )