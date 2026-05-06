import sys
import threading
import os
from pathlib import Path
from flask import Flask, render_template, request, jsonify, session, redirect, url_for, flash
from werkzeug.security import generate_password_hash, check_password_hash

# Adiciona diretório raiz ao path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from common.constants import DEFAULT_CONFIDENCE, DEFAULT_LINE_Y_RATIO, DEFAULT_MIN_LABEL_VOTES
from common.model_utils import resolve_best_model
from camera.detector import FoodDetector

# Adiciona o diretorio Back-end ao path para importar o banco de dados
sys.path.insert(0, str(ROOT.parent / "Back-end"))
try:
    import database
except ImportError as e:
    print(f"Aviso: Não foi possível importar o banco de dados: {e}")
    database = None

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
    session.clear()
    return redirect(url_for('login'))

from flask import Response

@app.route('/video_feed')
def video_feed():
    if 'user_id' not in session:
        return "Unauthorized", 401
    
    mode = request.args.get('mode', 'live')
    return Response(detector.generate_webcam_frames(
        camera_id=0,
        mode=mode,
        line_y_ratio=DEFAULT_LINE_Y_RATIO,
        min_label_votes=DEFAULT_MIN_LABEL_VOTES,
        user_id=session.get('user_id')
    ), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/camera', methods=['POST'])
def start_camera():
    # Agora a câmera é ativada nativamente no front-end ajustando o src da imagem.
    # Esta rota pode ser usada apenas para log ou preparações futuras se necessário.
    mode = request.json.get('mode', 'live')
    return jsonify({"status": f"Câmera configurada para modo {mode}. Iniciando stream no navegador."})

@app.route('/api/photo', methods=['POST'])
def start_photo():
    import time
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
        import time
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
    if 'user_id' not in session:
        return jsonify({"error": "Unauthorized"}), 401
    
    stats = database.get_dashboard_stats(session['user_id']) if database else None
    if stats:
        return jsonify(stats)
    return jsonify({"error": "Não foi possível carregar as estatísticas"}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000, use_reloader=False)
