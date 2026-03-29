import sys
from collections import Counter, defaultdict, deque
from pathlib import Path

import cv2
from ultralytics import YOLO

# Adiciona o diretório raiz ao path para importar common
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.constants import (
    CLASS_COLORS,
    DEFAULT_CAMERA_BUFFER_SIZE,
    DEFAULT_CAMERA_HEIGHT,
    DEFAULT_CAMERA_WIDTH,
    DISPLAY_NAMES,
)


class FoodDetector:
    """
    Detector de alimentos usando YOLO para identificação em tempo real.
    
    Suporta detecção em imagens estáticas e streaming de vídeo com dois modos:
    - 'live': Detecção frame a frame sem contagem acumulada
    - 'conveyor': Modo esteira com tracking e contagem de objetos cruzando uma linha
    
    Args:
        model_path: Caminho para o arquivo .pt do modelo YOLO treinado
        conf: Threshold de confiança mínimo para detecções (0.0 a 1.0)
    """
    def __init__(self, model_path: str, conf: float = 0.7):
        self.model = YOLO(model_path)
        self.conf = conf

    def _label(self, class_id: int) -> str:
        raw = self.model.names.get(class_id, str(class_id))
        return DISPLAY_NAMES.get(raw, raw)

    def _count(self, result) -> Counter:
        counts = Counter()
        if result.boxes is None:
            return counts
        for box in result.boxes:
            class_id = int(box.cls[0])
            counts[self._label(class_id)] += 1
        return counts

    def _draw_counts(self, frame, counts: Counter):
        panel_x, panel_y = 18, 16
        panel_w, panel_h = 330, 180
        self._draw_panel(frame, panel_x, panel_y, panel_w, panel_h)

        cv2.putText(
            frame,
            "Contagem em tempo real",
            (panel_x + 14, panel_y + 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        y = panel_y + 70
        for name in ["Feijao", "Macarrao", "Arroz"]:
            self._draw_counter_row(frame, panel_x + 14, y, name, counts.get(name, 0))
            y += 34

        cv2.putText(
            frame,
            f"Total no frame: {sum(counts.values())}",
            (panel_x + 14, panel_y + panel_h - 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )
        return frame

    def _draw_panel(self, frame, x: int, y: int, w: int, h: int) -> None:
        overlay = frame.copy()
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (18, 24, 32), -1)
        cv2.addWeighted(overlay, 0.62, frame, 0.38, 0, frame)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (120, 150, 180), 1)

    def _draw_counter_row(self, frame, x: int, y: int, label: str, value: int) -> None:
        color = CLASS_COLORS.get(label, (200, 200, 200))
        cv2.circle(frame, (x + 7, y - 4), 7, color, -1)
        cv2.putText(
            frame,
            label,
            (x + 22, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.68,
            (240, 240, 240),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            str(value),
            (x + 200, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            color,
            2,
            cv2.LINE_AA,
        )

    def _draw_conveyor_overlay(self, frame, totals: Counter, line_y: int):
        h, w = frame.shape[:2]
        cv2.line(frame, (0, line_y), (w, line_y), (0, 230, 255), 2)
        cv2.putText(
            frame,
            "Linha de contagem",
            (20, max(20, line_y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 230, 255),
            2,
            cv2.LINE_AA,
        )

        panel_x, panel_y = 18, 16
        panel_w, panel_h = 360, 195
        self._draw_panel(frame, panel_x, panel_y, panel_w, panel_h)

        cv2.putText(
            frame,
            "Contagem acumulada (esteira)",
            (panel_x + 14, panel_y + 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        y = panel_y + 72
        for name in ["Feijao", "Macarrao", "Arroz"]:
            self._draw_counter_row(frame, panel_x + 14, y, name, totals.get(name, 0))
            y += 34

        cv2.putText(
            frame,
            f"Qtd total: {sum(totals.values())}",
            (panel_x + 14, panel_y + panel_h - 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.66,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )
        return frame

    def _stable_label(self, labels: deque, min_votes: int) -> str | None:
        if not labels:
            return None
        voted = Counter(labels).most_common(1)[0]
        if voted[1] >= min_votes:
            return voted[0]
        return None

    def predict_image(self, image_path: str) -> Counter:
        """
        Executa detecção em uma imagem estática e exibe o resultado.
        
        Args:
            image_path: Caminho para o arquivo de imagem
            
        Returns:
            Counter: Contagem de objetos detectados por classe
        """
        result = self.model.predict(source=image_path, conf=self.conf, verbose=False)[0]
        counts = self._count(result)
        frame = self._draw_counts(result.plot(), counts)
        print(dict(counts))
        cv2.imshow("Deteccao", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return counts

    def capture_and_predict_photo(self, camera_id: int = 0) -> Counter | None:
        """
        Captura uma foto da webcam e executa detecção.
        
        Args:
            camera_id: ID da câmera (0, 1, 2...)
            
        Returns:
            Counter | None: Contagem de objetos detectados ou None se falhar
        """
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            raise RuntimeError(f"Nao foi possivel abrir camera {camera_id}")

        ok, frame = cap.read()
        cap.release()
        if not ok:
            print("Nao foi possivel capturar foto da webcam.")
            return None

        photo_dir = Path(__file__).resolve().parent / "captures"
        photo_dir.mkdir(parents=True, exist_ok=True)
        photo_path = photo_dir / "captura_teste.jpg"
        cv2.imwrite(str(photo_path), frame)
        print(f"Foto capturada em: {photo_path}")

        return self.predict_image(str(photo_path))

    def predict_webcam(
        self,
        camera_id: int = 0,
        mode: str = "conveyor",
        line_y_ratio: float = 0.6,
        min_label_votes: int = 3,
    ):
        """
        Executa detecção em tempo real via webcam.
        
        Args:
            camera_id: ID da câmera (0, 1, 2...)
            mode: Modo de operação ('conveyor' para contagem com linha, 'live' para tempo real)
            line_y_ratio: Posição da linha de contagem no modo conveyor (0.0 no topo, 1.0 embaixo)
            min_label_votes: Número mínimo de frames para estabilizar a classe detectada
        
        Controles:
            - Pressione 'q' para sair
        """
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            raise RuntimeError(f"Nao foi possivel abrir camera {camera_id}")

        # Otimizações de performance
        cap.set(cv2.CAP_PROP_BUFFERSIZE, DEFAULT_CAMERA_BUFFER_SIZE)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, DEFAULT_CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, DEFAULT_CAMERA_HEIGHT)

        totals = Counter()
        track_labels = defaultdict(lambda: deque(maxlen=10))
        track_last_y = {}
        counted_ids = set()

        while True:
            ok, frame = cap.read()
            if not ok:
                continue

            if mode == "conveyor":
                result = self.model.track(
                    source=frame,
                    conf=self.conf,
                    persist=True,
                    verbose=False,
                    tracker="bytetrack.yaml",
                )[0]
            else:
                result = self.model.predict(source=frame, conf=self.conf, verbose=False)[0]

            counts = self._count(result)
            shown = result.plot()

            if mode == "conveyor":
                h = shown.shape[0]
                line_y = int(h * line_y_ratio)

                if result.boxes is not None and result.boxes.id is not None:
                    for box, track_id_tensor in zip(result.boxes, result.boxes.id):
                        track_id = int(track_id_tensor.item())
                        class_id = int(box.cls[0])
                        label = self._label(class_id)
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        cy = int((y1 + y2) / 2)

                        track_labels[track_id].append(label)
                        stable = self._stable_label(track_labels[track_id], min_label_votes)
                        prev_y = track_last_y.get(track_id)
                        track_last_y[track_id] = cy

                        # Conta apenas quando o centro cruza a linha de cima para baixo.
                        if (
                            stable is not None
                            and prev_y is not None
                            and track_id not in counted_ids
                            and prev_y < line_y <= cy
                        ):
                            totals[stable] += 1
                            counted_ids.add(track_id)

                shown = self._draw_conveyor_overlay(shown, totals, line_y)
            else:
                shown = self._draw_counts(shown, counts)

            cv2.imshow("Detector", shown)
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--source", default="webcam")
    parser.add_argument("--camera-id", type=int, default=0)
    parser.add_argument("--conf", type=float, default=0.7)
    parser.add_argument("--mode", choices=["conveyor", "live"], default="conveyor")
    parser.add_argument("--line-y", type=float, default=0.6)
    parser.add_argument("--min-label-votes", type=int, default=3)
    args = parser.parse_args()
    
    detector = FoodDetector(args.model, args.conf)
    if args.source.lower() == "webcam":
        detector.predict_webcam(
            camera_id=args.camera_id,
            mode=args.mode,
            line_y_ratio=args.line_y,
            min_label_votes=args.min_label_votes,
        )
    else:
        detector.predict_image(args.source)

