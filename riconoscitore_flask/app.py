from flask import Flask, jsonify, render_template, request
from objectRecinozer import ObjectRecognizer,ObjectRecognizerV2
from scraper import ScraperWiki
import os
import base64
from PIL import Image
import cv2
import io
from datetime import datetime



app = Flask(__name__)
print("Versione V1 del riconoscitore")
recognizer = ObjectRecognizer()
scraper = ScraperWiki()

# Assicura che la cartella static esista
os.makedirs('static', exist_ok=True)

def log(msg):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}")
    
def draw_bbox_on_image(image_path, bbox, label=None, output_path=None):
    """
    Disegna una bounding box sull'immagine e la salva.
    """
    image = cv2.imread(image_path)
    x, y, w, h = map(int, bbox)
    color = (0, 255, 0)  # verde
    thickness = 2

    cv2.rectangle(image, (x, y), (x + w, y + h), color, thickness)

    if label:
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        cv2.putText(image, label, (x, y - 10), font, font_scale, color, 2)

    if not output_path:
        output_path = image_path.replace(".jpg", "_bbox.jpg")
    cv2.imwrite(output_path, image)

    return output_path

def process_and_save_image(image_file, filename='upload.jpg'):
    """
    Processa e salva l'immagine ridimensionandola.
    """
    original_path = os.path.join('./static', filename)
    image_file.save(original_path)

    pil = Image.open(original_path)
    if pil.mode in ('RGBA', 'LA', 'P'):
        pil = pil.convert('RGB')

    pil.thumbnail((800, 600), Image.Resampling.LANCZOS)
    pil.save(original_path, 'JPEG', quality=85)
    return original_path

def img_to_base64(path):
    """
    Codifica immagine in base64 da file path.
    """
    with open(path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')

def get_enhanced_description(label):
    """
    Ottiene descrizione potenziata da scraper.
    """
    try:
        description = scraper.search(label)
        if len(description) < 100:
            enhanced = scraper.search(f"what is {label} definition")
            if len(enhanced) > len(description):
                description = enhanced
        return description
    except Exception as e:
        print(f"Errore nella ricerca: {e}")
        return f"Errore nel recupero delle informazioni per {label}: {str(e)}"

@app.route("/", methods=["GET", "POST"])
def index():
    label = description = None
    left_b64 = right_b64 = None
    left_raw_b64 = right_raw_b64 = None
    error = None

    if request.method == 'POST':
        if 'left_image' not in request.files or 'right_image' not in request.files:
            error = "Necessarie entrambe le immagini"
        else:
            left_p = process_and_save_image(request.files['left_image'], 'left_upload.jpg')
            right_p = process_and_save_image(request.files['right_image'], 'right_upload.jpg')

            # Ottieni base64 delle immagini raw (senza bounding box)
            left_raw_b64 = img_to_base64(left_p)
            right_raw_b64 = img_to_base64(right_p)

            label, raw_bbox = recognizer.recognize(left_p, right_p)

            if label and raw_bbox and len(raw_bbox) == 4:
                description = get_enhanced_description(label)
                left_out = draw_bbox_on_image(left_p, raw_bbox, label, output_path='static/result_left.jpg')
                right_out = draw_bbox_on_image(right_p, raw_bbox, label, output_path='static/result_right.jpg')
                left_b64 = img_to_base64(left_out)
                right_b64 = img_to_base64(right_out)
            else:
                error = "Riconoscimento fallito"

    return render_template('index.html',
                           label=label,
                           description=description,
                           left_img=left_b64,
                           right_img=right_b64,
                           left_raw_img=left_raw_b64,
                           right_raw_img=right_raw_b64,
                           error=error)

@app.route('/api/process', methods=['POST'])
def process_image():
    try:
        log("Ricevuta richiesta a /api/process")

        if not request.is_json:
            log("ERRORE: Content-Type non è application/json")
            return jsonify({"error": "Content-Type deve essere application/json"}), 400
        
        data = request.get_json()
        log("Dati JSON ricevuti")

        image_data_r = data.get('right_image')
        image_data_l = data.get('left_image')

        if not image_data_r or not image_data_l:
            log("ERRORE: 'right_image' o 'left_image' mancanti")
            return jsonify({"error": "Campi 'right_image' e 'left_image' richiesti"}), 400

        if not (image_data_r.startswith('data:image/') and image_data_l.startswith('data:image/')):
            log("ERRORE: Le immagini non sono in formato base64 valido")
            return jsonify({"error": "Immagini devono essere in formato base64 con prefisso data:image/"}), 400
        
        log("Immagini in formato base64 valide, inizio salvataggio...")

        # --- Decodifica e salva raw ---
        def save_b64(img_data, path):
            log(f"Salvataggio immagine in {path}")
            header, encoded = img_data.split(',', 1) if ',' in img_data else ("", img_data)
            img_bytes = base64.b64decode(encoded)
            with open(path, "wb") as f:
                f.write(img_bytes)
            return path
        
        save_dir = "saved_images"
        os.makedirs(save_dir, exist_ok=True)

        path_r = save_b64(image_data_r, os.path.join(save_dir, "right.jpg"))
        path_l = save_b64(image_data_l, os.path.join(save_dir, "left.jpg"))

        log(f"Immagini salvate: {path_r}, {path_l}")
        log("Inizio riconoscimento...")

        # --- Riconoscimento ---
        label, raw_bbox = recognizer.recognize(path_l, path_r)

        if not label or not raw_bbox or len(raw_bbox) != 4:
            log(f"ERRORE: Riconoscimento fallito - label: {label}, bbox: {raw_bbox}")
            return jsonify({"error": "Riconoscimento fallito"}), 500
        else:
            log(f"Oggetto riconosciuto: {label} con bounding box: {raw_bbox}")
            description = get_enhanced_description(label)
            log(f"Descrizione generata: {description}")

        # --- Applica bounding box prima di rispondere ---
        log("Disegno bounding box sull'immagine sinistra...")
        boxed_l = draw_bbox_on_image(path_l, raw_bbox, label, output_path=os.path.join(save_dir, "left_bbox.jpg"))
        log(f"Immagine con bounding box salvata: {boxed_l}")

        # --- Prepara i base64 per la risposta ---
        def img_file_to_b64(path):
            log(f"Codifica immagine in base64: {path}")
            with open(path, 'rb') as f:
                return "data:image/jpeg;base64," + base64.b64encode(f.read()).decode('utf-8')

        boxed_l_b64 = img_file_to_b64(boxed_l)

        # --- Costruisci risposta ---
        result = {
            "success": True,
            "message": "Immagine processata con successo",
            "label": label,
            "description": description,
            "image": boxed_l_b64,
        }

        log("Elaborazione completata con successo. Invio risposta.")
        return jsonify(result), 200

    except Exception as e:
        log(f"ECCEZIONE: Errore interno - {str(e)}")
        return jsonify({"error": f"Errore interno: {str(e)}"}), 500


@app.route('/ping', methods=['GET'])
def ping():
    print("📡 Ping ricevuto da Unity!")
    return jsonify({"status": "ok"}), 200


if __name__ == '__main__':
    print("🚀 Avvio del server Flask...")
    app.run(debug=True, host='0.0.0.0', port=5000)
