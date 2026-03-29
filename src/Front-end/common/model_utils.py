"""Utilitários para resolução e carregamento de modelos."""

from pathlib import Path


def resolve_best_model(custom_path: str | None = None) -> Path:
    """
    Resolve o caminho para o melhor modelo treinado.
    
    Args:
        custom_path: Caminho opcional para um modelo específico.
                    Se None, procura pelo modelo padrão em detector/runs/train/weights/
    
    Returns:
        Path: Caminho absoluto para o arquivo .pt do modelo
        
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

    # Procura pelo modelo padrão
    candidates = [
        root / "detector" / "runs" / "train" / "weights" / "best.pt",
        root / "detector" / "runs" / "train" / "weights" / "last.pt",
    ]

    for path in candidates:
        if path.exists():
            return path

    searched = "\n".join(str(p) for p in candidates)
    raise FileNotFoundError(
        f"Modelo treinado nao encontrado. Caminhos verificados:\n{searched}"
    )