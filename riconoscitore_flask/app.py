from flask import Flask, jsonify, render_template, request
from objectRecinozer import ObjectRecognizer,ObjectRecognizerV2
from scraper import ScraperWiki
import os
import base64
from PIL import Image
import cv2
import io

app = Flask(__name__)
recognizer = ObjectRecognizer()
scraper = ScraperWiki()

# Assicura che la cartella static esista
os.makedirs('static', exist_ok=True)

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
        # Controlla se la richiesta è JSON
        if not request.is_json:
            return jsonify({"error": "Content-Type deve essere application/json"}), 400
        
        data = request.get_json()
        
        # Estrai l'immagine (base64)
        image_data_r = data.get('right_image')
        image_data_l = data.get('left_image')
        if not image_data_r:
            return jsonify({"error": "Campo 'right_image' richiesto"}), 400
        
        if not image_data_l:
            return jsonify({"error": "Campo 'left_image' richiesto"}), 400
                
        # Valida il formato dell'immagine
        if not (image_data_r.startswith('data:image/') and image_data_l.startswith('data:image/')):
            return jsonify({"error": "Immagine deve essere in formato base64 con prefisso data:image/"}), 400
                
        # Log della richiesta
        print(f"📸 Ricevuta immagine, formato: {image_data_r[:30]}...")
        print(f"📸 Ricevuta immagine, formato: {image_data_l[:30]}...")
        # DECODIFICA IMMAGINE BASE64
        try:
            
            # Rimuovi il prefisso data:image/jpeg;base64, 
            if ',' in image_data_r and ',' in image_data_l:
                header, encoded_data_r = image_data_r.split(',', 1)
                header, encoded_data_l = image_data_l.split(',', 1)
            else:
                encoded_data_r = image_data_r
                encoded_data_l = image_data_l
            # Decodifica base64 in bytes
            image_bytes_r = base64.b64decode(encoded_data_r)
            print(f"✅ Immagine decodificata: {len(image_bytes_r)} bytes")
            image_bytes_l = base64.b64decode(encoded_data_l)
            print(f"✅ Immagine decodificata: {len(image_bytes_l)} bytes")
            # Cartella di destinazione
            save_dir = "saved_images"
            os.makedirs(save_dir, exist_ok=True)  # Crea la cartella se non esiste

            # Percorso completo del file da salvare
            save_path_r = os.path.join(save_dir, "right.jpg")
            save_path_l = os.path.join(save_dir, "left.jpg")
            # Scrive l'immagine su disco
            with open(save_path_r, "wb") as f:
                f.write(image_bytes_r)
            with open(save_path_l, "wb") as f:
                f.write(image_bytes_l)

            print(f"💾 Immagine destra salvata in: {save_path_r}")
            print(f"💾 Immagine comunista salvata in: {save_path_l}")
            
            label, raw_bbox = recognizer.recognize(save_path_l, save_path_r)
            if label:
                description = get_enhanced_description(label)
            
        except Exception as decode_error:
            return jsonify({"error": f"Errore decodifica immagine: {str(decode_error)}"}), 400
        
        result = {
            "success": True,
            "message": "Immagine processata con successo",
            "label": label,
            "description": description,  # Immagine processata in base64
        }
        
        return jsonify(result), 200
        
    except Exception as e:
        return jsonify({"error": f"Errore interno: {str(e)}"}), 500


if __name__ == '__main__':
    print("🚀 Avvio del server Flask...")
    app.run(debug=True, host='0.0.0.0', port=5000)
