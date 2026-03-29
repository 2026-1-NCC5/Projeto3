#!/usr/bin/env python3
"""
Camera Detection Script - Para testes rápidos e desenvolvimento

Este script é para fins de desenvolvimento ou testes rápidos.
Para usar a interface completa com menu, execute:
    python main.py

Para argumentos:
    python run.py --help
"""

import argparse
import sys
from pathlib import Path

# Adiciona o diretório raiz ao path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.constants import (
    DEFAULT_CONFIDENCE,
    DEFAULT_LINE_Y_RATIO,
    DEFAULT_MIN_LABEL_VOTES,
)
from common.model_utils import resolve_best_model
from camera.detector import FoodDetector


def parse_args():
    parser = argparse.ArgumentParser(
        description="Script de teste rápido - Detector de Alimentos",
        epilog="""
Exemplos:
  python run.py                          # Mode conveyor (padrão)
  python run.py --mode live              # Mode live
  python run.py --camera-id 1            # Usa câmera 1
  python run.py --conf 0.5               # Confiança 0.5
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--model",
        default=None,
        help="Caminho opcional para um .pt específico",
    )
    parser.add_argument("--camera-id", type=int, default=0, help="ID da câmera (padrão: 0)")
    parser.add_argument("--conf", type=float, default=DEFAULT_CONFIDENCE, help="Confiança mínima")
    parser.add_argument(
        "--mode",
        choices=["conveyor", "live"],
        default="conveyor",
        help="Modo de detecção",
    )
    parser.add_argument("--line-y", type=float, default=DEFAULT_LINE_Y_RATIO, help="Linha Y")
    parser.add_argument(
        "--min-label-votes", type=int, default=DEFAULT_MIN_LABEL_VOTES, help="Votos mínimos"
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("┌─────────────────────────────────────────────────────────────┐")
    print("│ 📹 Detector de Alimentos - Script de Teste                  │")
    print("└─────────────────────────────────────────────────────────────┘\n")
    
    # Carregar modelo
    model_path = resolve_best_model(args.model)
    print(f"📦 Modelo: {model_path}")
    
    detector = FoodDetector(str(model_path), conf=args.conf)
    
    print(f"""
🚀 Iniciando Detector
   ├─ 📹 Câmera: {args.camera_id}
   ├─ 🎯 Modo: {args.mode}
   ├─ ⚙️  Confiança: {args.conf}
   └─ 🔧 Line Y: {args.line_y}

❓ Controles:
   • Pressione 'q' ou 'ESC' para sair
   • Pressione 'r' para resetar (modo conveyor)

""")
    
    # Executar detecção
    detector.predict_webcam(
        camera_id=args.camera_id,
        mode=args.mode,
        line_y_ratio=args.line_y,
        min_label_votes=args.min_label_votes,
    )
    
    print("\n✓ Encerrando.\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⏹️  Interrupção do usuário.\n")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Erro: {e}\n")
        sys.exit(1)
