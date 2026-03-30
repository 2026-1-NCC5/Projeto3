#!/usr/bin/env python3
"""
🍽️  Detector de Alimentos - Interface Principal

Menu interativo com suporte a:
- 📹 Detecção em tempo real com webcam
- 📷 Captura de foto via webcam
- 🖼️  Análise de arquivo salvo

Uso:
  python main.py
"""

import sys
import cv2
from pathlib import Path

# Adiciona diretório raiz ao path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from common.constants import (
    DEFAULT_CONFIDENCE,
    DEFAULT_LINE_Y_RATIO,
    DEFAULT_MIN_LABEL_VOTES,
)
from common.model_utils import resolve_best_model
from camera.detector import FoodDetector


# ─────────────────────────────────────────────────────────────────────────
# MENU E INTERAÇÃO
# ─────────────────────────────────────────────────────────────────────────

MAIN_MENU = """
╔════════════════════════════════════════════════════════════════╗
║           🍽️  DETECTOR DE ALIMENTOS - MENU PRINCIPAL           ║
╚════════════════════════════════════════════════════════════════╝

Selecione uma opção:

    0️⃣  📹 Camera (detecção em tempo real)
    1️⃣  📷 Foto (captura via webcam)
    2️⃣  🖼️  Arquivo Salvo (imagem do disco)
    
    q️  ❌ Sair

"""


def print_menu():
    """Exibe o menu principal com formatação sofisticada."""
    print(MAIN_MENU)


def get_menu_choice() -> str:
    """Solicita escolha do usuário no menu."""
    return input("👉 Escolha uma opção (0/1/2/q): ").strip().lower()


def detect_available_cameras(max_cameras: int = 5) -> list[int]:
    """Detecta webcams disponíveis testando IDs."""
    available = []
    for i in range(max_cameras):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available.append(i)
            cap.release()
    return available


def ask_camera_id() -> int:
    """Pergunta qual câmera usar com formatação melhorada."""
    print("\n🔍 Detectando webcams disponiveis...")
    available = detect_available_cameras()
    
    if not available:
        print("❌ Nenhuma webcam detectada!")
        print("Usando camera ID 0 (padrão)...")
        return 0
    
    print(f"✅ Webcams encontradas: {', '.join(map(str, available))}")
    
    while True:
        try:
            choice = input(f"\n📹 Escolha a câmera [{available[0]}]: ").strip()
            
            if not choice:
                return available[0]
            
            camera_id = int(choice)
            if camera_id in available:
                return camera_id
            else:
                print(f"❌ Câmera {camera_id} não disponível. Escolha entre: {available}")
        except ValueError:
            print("❌ Digite um número válido!")


def ask_detection_mode() -> str:
    """Pergunta o modo de detecção (conveyor ou live)."""
    print("\n═══════════════════════════════════════")
    print("    Escolha o modo de detecção:")
    print("═══════════════════════════════════════")
    print("  1️⃣  Conveyor (esteira com contagem)")
    print("  2️⃣  Live (detecção simples)")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    
    choice = input("👉 Escolha [1]: ").strip()
    
    if choice == "2":
        return "live"
    return "conveyor"


def ask_image_path() -> Path | None:
    """Solicita caminho de imagem com validação."""
    print()
    raw = input("🖼️  Informe o caminho da imagem (ou deixar em branco para cancelar): ").strip()
    
    if not raw:
        return None
    
    candidate = Path(raw)
    if not candidate.is_absolute():
        candidate = ROOT / candidate
    
    if not candidate.exists() or not candidate.is_file():
        print(f"❌ Arquivo inválido: {candidate}")
        return None
    
    return candidate


def display_detection_results(counts: dict) -> None:
    """Exibe resultados da detecção com formatação."""
    if not counts:
        print("\n📊 Resultado: Nenhum alimento detectado.")
        return
    
    top_label = max(counts, key=counts.get)
    top_count = counts[top_label]
    
    print("\n╔════════════════════════════════════════╗")
    print(f"║  📊 Detecção Concluída                 ║")
    print(f"║  Principal: {top_label:.<21} {top_count:>2} ║")
    print("╚════════════════════════════════════════╝")
    print("\n📈 Detalhes:")
    for label, count in sorted(counts.items(), key=lambda x: x[1], reverse=True):
        print(f"   {label}: {count}")


def run_image_analysis_loop(detector: FoodDetector) -> None:
    """Loop para analisar múltiplas imagens."""
    while True:
        image_path = ask_image_path()
        
        if image_path is None:
            back = input("\n⬅️  Voltar ao menu? [S/n]: ").strip().lower()
            if back in {"s", "sim", "", "y", "yes"}:
                return
            continue
        
        print(f"\n⏳ Analisando: {image_path.name}...")
        counts = detector.predict_image(str(image_path))
        display_detection_results(dict(counts))
        
        action = input("\n1️⃣  Analisar outra | 0️⃣  Menu principal: ").strip()
        if action == "0":
            return


def run_camera_detection(detector: FoodDetector, conf: float, line_y: float, min_votes: int):
    """Executa detecção em tempo real com câmera."""
    camera_id = ask_camera_id()
    detection_mode = ask_detection_mode()
    
    print(f"\n🚀 Iniciando detecção...")
    print(f"   📹 Câmera: {camera_id}")
    print(f"   🎯 Modo: {detection_mode}")
    print(f"   ⚙️  Confiança: {conf}")
    print(f"\n   ❓ Controles:")
    print(f"     • Pressione 'q' ou 'ESC' para sair")
    if detection_mode == "conveyor":
        print(f"     • Pressione 'r' para resetar contadores")
    print()
    
    detector.predict_webcam(
        camera_id=camera_id,
        mode=detection_mode,
        line_y_ratio=line_y,
        min_label_votes=min_votes,
    )


def run_photo_capture(detector: FoodDetector):
    """Captura foto via webcam e analisa."""
    camera_id = ask_camera_id()
    
    print(f"\n📷 Iniciando captura...")
    print(f"   Câmera: {camera_id}")
    print(f"   Pressione SPACE ou ENTER para capturar")
    print(f"   Pressione ESC para cancelar\n")
    
    detector.capture_and_predict_photo(camera_id=camera_id)


# ─────────────────────────────────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────────────────────────────────

def main():
    """Menu principal interativo."""
    model_path = resolve_best_model(None)
    
    print("┌────────────────────────────────────────────────────────────────┐")
    print("│ 🔧 Carregando modelo...                                        │")
    print("└────────────────────────────────────────────────────────────────┘")
    print(f"📦 Modelo: {model_path}\n")
    
    detector = FoodDetector(str(model_path), conf=DEFAULT_CONFIDENCE)
    
    print("✅ Modelo carregado com sucesso!\n")
    
    while True:
        print_menu()
        choice = get_menu_choice()
        
        if choice == "q":
            print("\n👋 Encerrando. Até logo!\n")
            return
        
        elif choice == "0":
            run_camera_detection(
                detector,
                conf=DEFAULT_CONFIDENCE,
                line_y=DEFAULT_LINE_Y_RATIO,
                min_votes=DEFAULT_MIN_LABEL_VOTES,
            )
        
        elif choice == "1":
            run_photo_capture(detector)
        
        elif choice == "2":
            run_image_analysis_loop(detector)
        
        else:
            print("\n⚠️  Opção inválida. Escolha 0, 1, 2 ou q.\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⏹️  Interrupção do usuário. Encerrando...\n")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Erro: {e}\n")
        sys.exit(1)
