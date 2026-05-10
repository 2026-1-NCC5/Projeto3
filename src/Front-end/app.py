import sys
import threading
import os
import time
import urllib.parse
from pathlib import Path
from flask import Flask, render_template, request, jsonify, session, redirect, url_for, flash
from werkzeug.security import generate_password_hash, check_password_hash

# Adiciona diretório raiz ao path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from common.constants import DEFAULT_CONFIDENCE, DEFAULT_LINE_Y_RATIO, DEFAULT_MIN_LABEL_VOTES
from common.model_utils import resolve_best_model
from camera.detector import (
    FoodDetector,
    set_camera_stream_active,
)

# Adiciona o diretorio Back-end ao path para importar o banco de dados
sys.path.insert(0, str(ROOT.parent / "Back-end"))
try:
    import database
except ImportError as e:
    print(f"Aviso: Não foi possível importar o banco de dados: {e}")
    database = None

if database is not None:
    add_pending_camera_item = database.add_pending_camera_item
    delete_pending_camera_item = database.delete_pending_camera_item
    clear_pending_camera_batch = database.clear_pending_camera_batch
    get_pending_camera_batch = database.get_pending_camera_batch
    update_user = database.update_user
else:
    def add_pending_camera_item(*args, **kwargs):
        return None

    def clear_pending_camera_batch(*args, **kwargs):
        return None

    def get_pending_camera_batch(*args, **kwargs):
        return {"items": [], "total_quantity": 0, "total_weight_kg": 0.0}

app = Flask(__name__)
app.secret_key = "super_secret_foodsteward_key_123"

# Carrega o modelo globalmente
print("Iniciando servidor web...")
model_path = resolve_best_model(None)
print(f"📦 Modelo carregado: {model_path}")
detector = FoodDetector(str(model_path), conf=DEFAULT_CONFIDENCE)

@app.route('/')
def index():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template('index.html', username=session.get('username'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        user = database.get_user_by_username(username) if database else None
        if user and check_password_hash(user['password_hash'], password):
            session['user_id'] = user['id']
            session['username'] = user['username']
            return redirect(url_for('index'))
        else:
            flash("Usuário ou senha incorretos.")
            
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        if not username or not password:
            flash("Preencha todos os campos.")
        else:
            hashed_pw = generate_password_hash(password)
            success = database.create_user(username, hashed_pw) if database else False
            if success:
                flash("Conta criada com sucesso! Faça login.")
                return redirect(url_for('login'))
            else:
                flash("Este nome de usuário já existe.")
                
    return render_template('register.html')

@app.route('/logout')
def logout():
    user_id = session.get('user_id')
    session_key = session.get('camera_session_key')
    if user_id is not None:
        set_camera_stream_active(user_id, False)
    if session_key:
        clear_pending_camera_batch(session_key, user_id)
    session.clear()
    return redirect(url_for('login'))

from flask import Response

@app.route('/api/cameras')
def list_cameras():
    """Enumera câmeras disponíveis tentando abrir índices 0-5."""
    import cv2
    cameras = []
    for idx in range(6):
        cap = cv2.VideoCapture(idx, cv2.CAP_AVFOUNDATION)
        if not cap.isOpened():
            cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            cameras.append({"id": idx, "label": f"C\u00e2mera {idx}" if idx > 0 else f"C\u00e2mera Padr\u00e3o ({idx})"})
            cap.release()
        else:
            if cap:
                cap.release()
    if not cameras:
        cameras = [{"id": 0, "label": "C\u00e2mera Padr\u00e3o (0)"}]
    return jsonify({"cameras": cameras})


@app.route('/video_feed')
def video_feed():
    if 'user_id' not in session:
        return "Unauthorized", 401
    
    mode = request.args.get('mode', 'live')
    roi_size = int(request.args.get('roi_size', 80))
    primary_color = urllib.parse.unquote(request.args.get('color', '#1A4D2E'))
    camera_id = int(request.args.get('camera_id', 0))
    user_id = session.get('user_id')
    
    # 1. Força a parada de qualquer stream anterior para liberar o hardware
    set_camera_stream_active(user_id, False)
    time.sleep(0.3) # Pequena pausa para o loop anterior detectar a mudança
    set_camera_stream_active(user_id, True)
    
    session_key = session.get('camera_session_key')
    if not session_key:
        session_key = os.urandom(16).hex()
        session['camera_session_key'] = session_key
    return Response(detector.generate_webcam_frames(
        camera_id=camera_id,
        mode=mode,
        roi_size=roi_size,
        primary_color=primary_color,
        line_y_ratio=DEFAULT_LINE_Y_RATIO,
        min_label_votes=1,
        user_id=session.get('user_id'),
        session_key=session_key,
    ), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/user/update', methods=['POST'])
def user_update():
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({"error": "Unauthorized"}), 401
        
    username = request.json.get('username')
    password = request.json.get('password')
    
    password_hash = None
    if password:
        password_hash = generate_password_hash(password)
        
    success = update_user(user_id, username, password_hash)
    if success:
        if username:
            session['username'] = username
        return jsonify({"status": "Perfil atualizado com sucesso"})
    return jsonify({"error": "Erro ao atualizar perfil"}), 500

@app.route('/api/camera/stop', methods=['POST'])
def stop_camera():
    if 'user_id' not in session:
        return jsonify({"error": "Unauthorized"}), 401

    set_camera_stream_active(session.get('user_id'), False)
    return jsonify({"status": "Câmera encerrada com sucesso."})


@app.route('/api/camera/history')
def camera_history():
    if 'user_id' not in session:
        return jsonify({"error": "Unauthorized"}), 401

    session_key = session.get('camera_session_key')
    if not session_key:
        response = jsonify({"items": [], "total_quantity": 0, "total_weight_kg": 0.0})
        response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
        return response

    response = jsonify(get_pending_camera_batch(session_key, session.get('user_id')))
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    return response


@app.route('/api/camera/delete_item', methods=['POST'])
def delete_item():
    session_key = session.get('camera_session_key')
    user_id = session.get('user_id')
    name = request.json.get('name')
    
    if not session_key or not user_id or not name:
        return jsonify({"error": "Dados insuficientes"}), 400
        
    success = delete_pending_camera_item(session_key, user_id, name)
    if success:
        return jsonify({"status": "Item removido com sucesso"})
    return jsonify({"error": "Erro ao remover item"}), 500

@app.route('/api/camera/confirm', methods=['POST'])
def confirm_camera_batch():
    if 'user_id' not in session:
        return jsonify({"error": "Unauthorized"}), 401

    user_id = session.get('user_id')
    session_key = session.get('camera_session_key')
    if not session_key:
        return jsonify({"error": "Nenhuma sessão de câmera ativa."}), 400

    batch = get_pending_camera_batch(session_key, user_id)

    if not batch["items"]:
        return jsonify({"error": "Nenhum item pendente para confirmar."}), 400

    if not database:
        return jsonify({"error": "Banco de dados indisponível."}), 500

    for item in batch["items"]:
        quantity = int(item.get("quantity", 0))
        confidence = item.get("confidence")
        for _ in range(quantity):
            database.add_food_item(item["name"], None, confidence, user_id)

    clear_pending_camera_batch(session_key, user_id)
    return jsonify({"status": "success", "persisted": batch})

@app.route('/api/camera', methods=['POST'])
def start_camera():
    mode = request.json.get('mode', 'live')
    force_reset = request.json.get('force_reset', False)
    
    session_key = session.get('camera_session_key')
    
    if not session_key or force_reset:
        session_key = os.urandom(16).hex()
        session['camera_session_key'] = session_key
        clear_pending_camera_batch(session_key, session.get('user_id'))
        status = f"Nova sessão iniciada para modo {mode}."
    else:
        status = f"Continuando sessão existente para modo {mode}."
        
    return jsonify({"status": status, "session_key": session_key})

@app.route('/api/photo', methods=['POST'])
def start_photo():
    result_filename = f"photo_{int(time.time())}.jpg"
    result_path = ROOT / "static" / "results" / result_filename
    (ROOT / "static" / "results").mkdir(parents=True, exist_ok=True)
    
    try:
        counts = detector.capture_and_predict_photo(
            camera_id=0, 
            show=False, 
            save_path=str(result_path),
            user_id=session.get('user_id')
        )
        if counts is not None:
            return jsonify({
                "status": "success",
                "results": dict(counts),
                "image_url": url_for('static', filename=f"results/{result_filename}")
            })
        else:
            return jsonify({"error": "Falha na captura"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/analyze', methods=['POST'])
def analyze_file():
    if 'file' not in request.files:
        return jsonify({"error": "Nenhum arquivo enviado"}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "Arquivo vazio"}), 400
        
    if file:
        # Salva o arquivo temporariamente
        temp_path = ROOT / "temp_upload.jpg"
        file.save(str(temp_path))
        
        # Analisa e salva a imagem com boxes
        result_filename = f"result_{int(time.time())}.jpg"
        result_path = ROOT / "static" / "results" / result_filename
        (ROOT / "static" / "results").mkdir(parents=True, exist_ok=True)
        
        counts = detector.predict_image(
            str(temp_path), 
            show=False, 
            save_path=str(result_path),
            user_id=session.get('user_id')
        )
        
        # Remove o arquivo temporário
        if temp_path.exists():
            os.remove(temp_path)
            
        return jsonify({
            "results": dict(counts),
            "image_url": url_for('static', filename=f"results/{result_filename}")
        })

@app.route('/api/stats')
def get_stats():
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({"error": "Unauthorized"}), 401
    
    name_filter = request.args.get('name')
    date_from = request.args.get('date_from')
    
    stats = database.get_dashboard_stats(user_id, name_filter, date_from) if database else None
    if stats:
        return jsonify(stats)
    return jsonify({"error": "Não foi possível carregar as estatísticas"}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000, use_reloader=False, threaded=True)
