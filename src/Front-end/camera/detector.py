import sys
import threading
import time
import numpy as np
from collections import Counter, defaultdict, deque
from pathlib import Path

import cv2
from ultralytics import YOLO

# Adiciona o diretório raiz ao path para importar common
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Adiciona o diretorio Back-end ao path para importar o banco de dados
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "Back-end"))
try:
    from database import add_food_item, add_pending_camera_item
except ImportError as e:
    print(f"Aviso: Não foi possível importar o banco de dados: {e}")
    def add_food_item(name, track_id, confidence=None, user_id=None): pass
    def add_pending_camera_item(session_key, user_id, name, quantity=1, weight_kg=0.0, confidence=None): pass

from common.constants import (
    CLASS_COLORS,
    DEFAULT_CAMERA_BUFFER_SIZE,
    DEFAULT_CAMERA_HEIGHT,
    DEFAULT_CAMERA_WIDTH,
    ESTIMATED_ITEM_WEIGHT_KG,
    DISPLAY_NAMES,
    RICE_WEIGHT_THRESHOLDS,
)


_ACTIVE_STREAMS = {}
_ACTIVE_STREAMS_LOCK = threading.Lock()


def set_camera_stream_active(user_id, active: bool) -> None:
    if user_id is None:
        return
    with _ACTIVE_STREAMS_LOCK:
        if active:
            _ACTIVE_STREAMS[user_id] = True
        else:
            _ACTIVE_STREAMS.pop(user_id, None)


def is_camera_stream_active(user_id) -> bool:
    if user_id is None:
        return True
    with _ACTIVE_STREAMS_LOCK:
        return _ACTIVE_STREAMS.get(user_id, False)


def try_open_camera(camera_id, retries=5, skip_frames=5):
    cap = None
    # Força garbage collection para liberar câmeras prévias no macOS
    import gc
    gc.collect()
    time.sleep(0.2)
    
    for i in range(retries):
        # Tenta AVFoundation no Mac, depois fallback padrão
        for cid in [camera_id]:
            try:
                # macOS com AVFoundation geralmente é mais estável
                try:
                    cap = cv2.VideoCapture(cid, cv2.CAP_AVFOUNDATION)
                except:
                    cap = cv2.VideoCapture(cid)
                    
                if cap.isOpened():
                    # Libera buffer para evitar frames antigos
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    # Skip inicial de frames para garantir frame fresco
                    for _ in range(skip_frames):
                        ret, _ = cap.read()
                        if not ret:
                            cap.release()
                            cap = None
                            break
                    if cap:
                        return cap
                if cap: 
                    cap.release()
                    del cap
            except Exception as e:
                if cap: 
                    cap.release()
                    del cap
                print(f"Erro ao abrir câmera {cid}: {e}")
        time.sleep(0.7)  # Aumentado para dar mais tempo de liberação de hardware
    return None


class FoodDetector:
    def __init__(self, model_path: str, conf: float = 0.7):
        self.model = YOLO(model_path)
        self.conf = conf
        self._infer_lock = threading.Lock()

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
        return frame # Dashboard lida com visualização

    def _stable_label(self, labels: deque, min_votes: int) -> str | None:
        if not labels: return None
        voted = Counter(labels).most_common(1)[0]
        return voted[0] if voted[1] >= min_votes else None

    def predict_image(self, image_path: str, show: bool = True, save_path: str = None, user_id: int = None) -> Counter:
        result = self.model.predict(source=image_path, conf=self.conf, verbose=False)[0]
        counts = self._count(result)
        frame = result.plot()
        if result.boxes is not None:
            for box in result.boxes:
                class_id = int(box.cls[0])
                label = self._label(class_id)
                conf = float(box.conf[0].item())
                add_food_item(label, None, conf, user_id)
        if save_path: cv2.imwrite(save_path, frame)
        return counts

    def capture_and_predict_photo(self, camera_id: int = 0, show: bool = True, save_path: str = None, user_id: int = None) -> Counter | None:
        cap = try_open_camera(camera_id, retries=5, skip_frames=2)
        if not cap: return None
        ok, frame = cap.read()
        cap.release()
        if not ok: return None
        
        photo_dir = Path(__file__).resolve().parent / "captures"
        photo_dir.mkdir(parents=True, exist_ok=True)
        photo_path = photo_dir / f"cap_{int(time.time())}.jpg"
        cv2.imwrite(str(photo_path), frame)
        return self.predict_image(str(photo_path), show=show, save_path=save_path, user_id=user_id)

    def generate_webcam_frames(self, camera_id=0, mode="conveyor", roi_size=80, primary_color="#1A4D2E", line_y_ratio=0.6, min_label_votes=3, user_id=None, session_key=None):
        cap = try_open_camera(camera_id, retries=5, skip_frames=5)
        if not cap:
            err = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(err, "ERRO: CAMERA NAO ENCONTRADA", (100, 240), 1, 1.5, (0,0,255), 2)
            ret, buf = cv2.imencode('.jpg', err)
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')
            return

        try:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            totals = Counter()
            track_labels = defaultdict(lambda: deque(maxlen=10))
            track_last_y = {}
            counted_ids = set()
            last_frame_time = 0

            while is_camera_stream_active(user_id):
                if time.time() - last_frame_time < 0.04:
                    time.sleep(0.01)
                    continue
                
                ok, frame = cap.read()
                if not ok:
                    time.sleep(0.1)
                    continue
                
                last_frame_time = time.time()
                h, w = frame.shape[:2]
                roi_px = int(min(w, h) * (roi_size / 100))
                rx1, ry1 = (w - roi_px) // 2, (h - roi_px) // 2
                rx2, ry2 = rx1 + roi_px, ry1 + roi_px
                line_y = ry1 + int((ry2 - ry1) * line_y_ratio)

                try:
                    p_bgr = tuple(int(primary_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))[::-1]
                except Exception:
                    p_bgr = (0, 255, 0)

                if mode == "snapshot":
                    # Desenha a ROI no frame de preview para o usuário se posicionar
                    cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), (0, 0, 0), 3)
                    cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), p_bgr, 2)
                    cv2.putText(frame, "POSICIONE O ALIMENTO AQUI", (rx1, ry1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, p_bgr, 2)
                    
                    ret, buf = cv2.imencode('.jpg', frame)
                    yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')
                    continue

                try:
                    with self._infer_lock:
                        result = self.model.track(frame, conf=self.conf, persist=True, verbose=False)[0]
                        shown = result.plot()

                    # --- DESENHO DA ZONA DE CALIBRAÇÃO (ROI) ---
                    # Desenha o Quadrado da ROI
                    cv2.rectangle(shown, (rx1, ry1), (rx2, ry2), (0, 0, 0), 3) # Borda preta externa
                    cv2.rectangle(shown, (rx1, ry1), (rx2, ry2), p_bgr, 2) # Linha da cor principal
                    
                    # Desenha Cantos Reforçados (Glow effect)
                    cl = 40 # Comprimento do canto
                    for offset, thickness, color in [(0, 5, (0,0,0)), (0, 2, p_bgr)]:
                        # Top Left
                        cv2.line(shown, (rx1, ry1), (rx1 + cl, ry1), color, thickness)
                        cv2.line(shown, (rx1, ry1), (rx1, ry1 + cl), color, thickness)
                        # Top Right
                        cv2.line(shown, (rx2, ry1), (rx2 - cl, ry1), color, thickness)
                        cv2.line(shown, (rx2, ry1), (rx2, ry1 + cl), color, thickness)
                        # Bottom Left
                        cv2.line(shown, (rx1, ry2), (rx1 + cl, ry2), color, thickness)
                        cv2.line(shown, (rx1, ry2), (rx1, ry2 - cl), color, thickness)
                        # Bottom Right
                        cv2.line(shown, (rx2, ry2), (rx2 - cl, ry2), color, thickness)
                        cv2.line(shown, (rx2, ry2), (rx2, ry2 - cl), color, thickness)

                    # Desenha a Linha de Sensor (apenas no modo conveyor)
                    if mode == "conveyor":
                        cv2.line(shown, (rx1, line_y), (rx2, line_y), (0, 0, 0), 4)
                        cv2.line(shown, (rx1, line_y), (rx2, line_y), p_bgr, 2)
                        cv2.putText(shown, "SENSOR", (rx1 + 5, line_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, p_bgr, 2)

                    if result.boxes is not None and result.boxes.id is not None:
                        for box, tid_tensor in zip(result.boxes, result.boxes.id):
                            tid = int(tid_tensor.item())
                            label = self._label(int(box.cls[0]))
                            conf = float(box.conf[0].item())
                            x1, y1, x2, y2 = box.xyxy[0].tolist()
                            cx, cy = (x1+x2)/2, (y1+y2)/2

                            if rx1 <= cx <= rx2 and ry1 <= cy <= ry2:
                                track_labels[tid].append(label)
                                stable = self._stable_label(track_labels[tid], min_label_votes)
                                prev_y = track_last_y.get(tid)
                                track_last_y[tid] = cy

                                if mode == "conveyor":
                                    if prev_y is not None and prev_y < line_y <= cy and tid not in counted_ids:
                                        # Cálculo de ratio (área do box vs área da ROI)
                                        ratio = ((x2-x1)*(y2-y1)) / (roi_px * roi_px)
                                        
                                        # Lógica específica para Arroz (1kg vs 5kg)
                                        if (stable or label) == "arroz":
                                            weight = 1.0
                                            for w_val, threshold in sorted(RICE_WEIGHT_THRESHOLDS.items(), reverse=True):
                                                if ratio >= threshold:
                                                    weight = w_val
                                                    break
                                        else:
                                            # Peso padrão para outros itens definido em constants.py
                                            weight = ESTIMATED_ITEM_WEIGHT_KG.get(stable or label, 1.0)
                                            
                                        totals[stable or label] += 1
                                        counted_ids.add(tid)
                                        if session_key: add_pending_camera_item(session_key, user_id, stable or label, 1, weight, conf)
                                else: # mode == live
                                    if tid not in counted_ids:
                                        ratio = ((x2-x1)*(y2-y1)) / (roi_px * roi_px)
                                        if (stable or label) == "arroz":
                                            weight = 1.0
                                            for w_val, threshold in sorted(RICE_WEIGHT_THRESHOLDS.items(), reverse=True):
                                                if ratio >= threshold:
                                                    weight = w_val
                                                    break
                                        else:
                                            weight = ESTIMATED_ITEM_WEIGHT_KG.get(stable or label, 1.0)
                                            
                                        totals[stable or label] += 1
                                        counted_ids.add(tid)
                                        if session_key: add_pending_camera_item(session_key, user_id, stable or label, 1, weight, conf)
                except Exception as e:
                    print(f"IA Error: {e}")
                    shown = frame

                cv2.putText(shown, f"SCANNER ATIVO: {roi_size}%", (rx1 + 10, ry1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                ret, buf = cv2.imencode('.jpg', shown)
                yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')
        finally:
            if cap: cap.release()

    def predict_webcam(self, camera_id=0, mode="conveyor", line_y_ratio=0.6, min_label_votes=3):
        # Implementação legada para CLI/Desktop se necessário
        cap = cv2.VideoCapture(camera_id)
        while True:
            ok, frame = cap.read()
            if not ok: break
            cv2.imshow("Detector", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    pass
