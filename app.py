from flask import Flask, render_template, request, jsonify, url_for, Response
import os
from werkzeug.utils import secure_filename
import cv2
import numpy as np
from ultralytics import YOLO
import base64
import json
from datetime import datetime
import random
import string
import logging
import time
import threading
import queue
import pymysql

# ------------------ Flask App Setup ------------------
app = Flask(__name__)

# ------------------ Configuration ------------------
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['CAPTURE_FOLDER'] = 'static/captures'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['CONF_THRESHOLD'] = 0.85
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# ------------------ Database Configuration ------------------
# Override via environment variables if needed
app.config['DB_HOST'] = os.getenv('DB_HOST', 'localhost')
app.config['DB_PORT'] = int(os.getenv('DB_PORT', '3306'))
app.config['DB_USER'] = os.getenv('DB_USER', 'root')
app.config['DB_PASSWORD'] = os.getenv('DB_PASSWORD', 'root')
app.config['DB_NAME'] = os.getenv('DB_NAME', 'dress')

def _get_db_connection():
    return pymysql.connect(
        host=app.config['DB_HOST'],
        port=app.config['DB_PORT'],
        user=app.config['DB_USER'],
        password=app.config['DB_PASSWORD'],
        database=app.config['DB_NAME'],
        cursorclass=pymysql.cursors.DictCursor,
        autocommit=True
    )

def _find_student_by_rfid(rfid_uid: str):
    if not rfid_uid:
        return None
    try:
        conn = _get_db_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT student_id, rfid_uid, name, year_level, course, college, photo
                    FROM students
                    WHERE rfid_uid = %s
                    LIMIT 1
                    """,
                    (rfid_uid,)
                )
                row = cur.fetchone()
                return row
        finally:
            conn.close()
    except Exception as e:
        app.logger.error(f"DB lookup error: {e}")
        return None

def _insert_rfid_log(rfid_uid: str, student_id: int | None, status: str):
    try:
        conn = _get_db_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO rfid_logs (student_id, rfid_uid, status)
                    VALUES (%s, %s, %s)
                    """,
                    (student_id, rfid_uid, status)
                )
        finally:
            conn.close()
    except Exception as e:
        app.logger.error(f"DB log insert error: {e}")

def _lookup_and_log(rfid_uid: str):
    student = _find_student_by_rfid(rfid_uid)
    if student:
        _insert_rfid_log(rfid_uid, student.get('student_id'), 'valid')
        return {
            'matched': True,
            'student_id': student.get('student_id'),
            'name': student.get('name'),
            'rfid_uid': student.get('rfid_uid'),
            'year_level': student.get('year_level'),
            'course': student.get('course'),
            'college': student.get('college'),
            'photo': student.get('photo'),
        }
    else:
        _insert_rfid_log(rfid_uid, None, 'unregistered')
        return {'matched': False}

# Ensure required folders exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['CAPTURE_FOLDER'], exist_ok=True)

# ------------------ Logging ------------------
logging.basicConfig(level=logging.INFO)
app.logger.setLevel(logging.INFO)

# ------------------ Load YOLO Model ------------------
model = YOLO('best.pt')
model.fuse()

# ------------------ Helper Functions ------------------
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def generate_filename(prefix='img', extension='jpg'):
    random_str = ''.join(random.choices(string.ascii_letters + string.digits, k=8))
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    return f'{prefix}_{timestamp}_{random_str}.{extension}'

# ------------------ RFID (ACR122U) Helpers ------------------
_rfid_io_lock = threading.Lock()

def get_rfid_uid(timeout_seconds=8):
    try:
        from smartcard.System import readers
        from smartcard.util import toHexString
        from smartcard.Exceptions import CardConnectionException
    except Exception as import_error:
        return None, f"pyscard not installed or unavailable: {import_error}"

    try:
        with _rfid_io_lock:
            available_readers = readers()
            if not available_readers:
                return None, 'No PC/SC readers found. Ensure ACR122U driver is installed.'

            reader = None
            # Prefer ACR122U if multiple
            for r in available_readers:
                if 'ACR122' in str(r).upper() or 'ACR 122' in str(r).upper():
                    reader = r
                    break
            if reader is None:
                reader = available_readers[0]

            connection = reader.createConnection()
            connection.connect()  # T=1 or direct

            # ACR122U: Get Card UID command
            get_uid_apdu = [0xFF, 0xCA, 0x00, 0x00, 0x00]

            # Poll until card present or timeout
            start_time = time.time()
            last_error = None
            while time.time() - start_time < timeout_seconds:
                try:
                    data, sw1, sw2 = connection.transmit(get_uid_apdu)
                    if sw1 == 0x90 and sw2 == 0x00 and data:
                        uid_hex = ''.join(f"{b:02X}" for b in data)
                        return uid_hex, None
                    last_error = f"Unexpected status: {sw1:02X}{sw2:02X}"
                except CardConnectionException as e:
                    last_error = str(e)
                time.sleep(0.25)

            return None, last_error or 'Timeout waiting for card'
    except Exception as e:
        return None, f"RFID error: {e}"


def read_mifare_classic_block(block_number, key_hex='FFFFFFFFFFFF', key_type='A', key_slot=0x00):
    try:
        from smartcard.System import readers
    except Exception as import_error:
        return None, f"pyscard not installed or unavailable: {import_error}"

    if key_type.upper() not in ('A', 'B'):
        return None, 'key_type must be A or B'

    try:
        key_bytes = [int(key_hex[i:i+2], 16) for i in range(0, 12, 2)]
    except Exception:
        return None, 'key_hex must be 12 hex chars (e.g., FFFFFFFFFFFF)'

    with _rfid_io_lock:
        available_readers = readers()
        if not available_readers:
            return None, 'No PC/SC readers found.'

        reader = None
        for r in available_readers:
            if 'ACR122' in str(r).upper() or 'ACR 122' in str(r).upper():
                reader = r
                break
        if reader is None:
            reader = available_readers[0]

        conn = reader.createConnection()
        conn.connect()

        def transmit(apdu):
            data, sw1, sw2 = conn.transmit(apdu)
            if not (sw1 == 0x90 and sw2 == 0x00):
                raise RuntimeError(f"APDU failed SW={sw1:02X}{sw2:02X}")
            return data

        # Load key into volatile slot
        load_key_apdu = [0xFF, 0x82, 0x00, key_slot, 0x06] + key_bytes
        transmit(load_key_apdu)

        # Authenticate
        key_code = 0x60 if key_type.upper() == 'A' else 0x61
        auth_apdu = [0xFF, 0x86, 0x00, 0x00, 0x05, 0x01, 0x00, block_number & 0xFF, key_code, key_slot]
        transmit(auth_apdu)

        # Read 16 bytes
        read_apdu = [0xFF, 0xB0, 0x00, block_number & 0xFF, 0x10]
        data = transmit(read_apdu)
        return bytes(data), None


def read_mifare_classic_range(start_block, num_blocks, key_hex='FFFFFFFFFFFF', key_type='A'):
    collected = bytearray()
    for b in range(start_block, start_block + num_blocks):
        data, err = read_mifare_classic_block(b, key_hex=key_hex, key_type=key_type)
        if err:
            return None, f"Block {b}: {err}"
        collected.extend(data)
    return bytes(collected), None

# Live RFID autoscan infrastructure
_rfid_thread = None
_rfid_thread_stop = threading.Event()
_rfid_subscribers = set()
_rfid_subscribers_lock = threading.Lock()
_rfid_last_uid = None
_rfid_last_time = 0.0
_rfid_presence_lock = threading.Lock()
_rfid_present = False

def _rfid_set_present(present: bool):
    global _rfid_present
    with _rfid_presence_lock:
        _rfid_present = bool(present)

def _rfid_is_present():
    with _rfid_presence_lock:
        return _rfid_present

def _publish_event(payload):
    with _rfid_subscribers_lock:
        for q in list(_rfid_subscribers):
            try:
                q.put_nowait(payload)
            except Exception:
                pass

def _rfid_poll_loop():
    global _rfid_last_uid, _rfid_last_time
    debounce_seconds = 2.0
    while not _rfid_thread_stop.is_set():
        uid, err = get_rfid_uid(timeout_seconds=1)
        now = time.time()
        if uid:
            if uid != _rfid_last_uid or (now - _rfid_last_time) > debounce_seconds:
                _rfid_last_uid = uid
                _rfid_last_time = now
                _rfid_set_present(True)
                match_info = _lookup_and_log(uid)
                _publish_event({'type': 'uid', 'uid': uid, 'match': match_info})
        else:
            # No UID read within this poll; mark not present
            _rfid_set_present(False)
        time.sleep(0.2)

def _ensure_rfid_thread_running():
    global _rfid_thread
    if _rfid_thread is None or not _rfid_thread.is_alive():
        _rfid_thread_stop.clear()
        _rfid_thread = threading.Thread(target=_rfid_poll_loop, name='rfid-poll', daemon=True)
        _rfid_thread.start()

# ------------------ Routes ------------------
@app.route('/')
def home():
    return render_template('dashboard.html')

@app.route('/rfid/read_uid', methods=['GET'])
def rfid_read_uid():
    try:
        _ensure_rfid_thread_running()
        uid, err = get_rfid_uid()
        if uid:
            _rfid_set_present(True)
            match_info = _lookup_and_log(uid)
            return jsonify({'success': True, 'uid': uid, 'match': match_info})
        _rfid_set_present(False)
        return jsonify({'success': False, 'error': err or 'Unknown error'}), 500
    except Exception as e:
        app.logger.error(f"RFID endpoint error: {e}")
        return jsonify({'success': False, 'error': 'Internal server error'}), 500

@app.route('/rfid/lookup', methods=['GET'])
def rfid_lookup():
    try:
        uid = request.args.get('uid')
        if not uid:
            return jsonify({'success': False, 'error': 'uid query param is required'}), 400
        student = _find_student_by_rfid(uid)
        if student:
            return jsonify({'success': True, 'uid': uid, 'student': student})
        return jsonify({'success': True, 'uid': uid, 'student': None})
    except Exception as e:
        app.logger.error(f"RFID lookup error: {e}")
        return jsonify({'success': False, 'error': 'Internal server error'}), 500

@app.route('/rfid/stream')
def rfid_stream():
    _ensure_rfid_thread_running()

    client_queue = queue.Queue(maxsize=10)
    with _rfid_subscribers_lock:
        _rfid_subscribers.add(client_queue)

    def gen():
        try:
            # Send a hello event; do not push any prior UID to avoid stale displays
            yield 'event: hello\ndata: rfid-stream-connected\n\n'
            while True:
                payload = client_queue.get()
                try:
                    yield 'data: ' + json.dumps(payload) + '\n\n'
                except Exception:
                    pass
        except GeneratorExit:
            pass
        finally:
            with _rfid_subscribers_lock:
                _rfid_subscribers.discard(client_queue)

    headers = {
        'Content-Type': 'text/event-stream',
        'Cache-Control': 'no-cache',
        'Connection': 'keep-alive',
        'X-Accel-Buffering': 'no'
    }
    return Response(gen(), headers=headers)

def _read_entire_ndef_area_default_key():
    # Scan blocks 4..62 (skip sector trailers)
    all_bytes = bytearray()
    for b in range(4, 63):
        # skip trailer blocks (7,11,15,...)
        if (b + 1) % 4 == 0:
            continue
        data, err = read_mifare_classic_block(b, key_hex='FFFFFFFFFFFF', key_type='A')
        if err or not data:
            # best-effort: continue; some sectors may be locked or use different keys
            continue
        all_bytes.extend(data)
    return bytes(all_bytes)

def _parse_ndef_text_from_tlv(tlv_bytes: bytes):
    # Look for NDEF TLV 0x03
    i = 0
    n = len(tlv_bytes)
    while i < n:
        t = tlv_bytes[i]
        i += 1
        if t == 0x00:  # NULL TLV
            continue
        if t == 0xFE:  # Terminator TLV
            break
        if i >= n:
            break
        length = tlv_bytes[i]
        i += 1
        if length == 0xFF:
            if i + 2 > n:
                break
            length = (tlv_bytes[i] << 8) | tlv_bytes[i + 1]
            i += 2
        if t != 0x03:  # not NDEF TLV
            i += length
            continue
        # NDEF message begins at i, length bytes long
        ndef = tlv_bytes[i:i+length]
        # Parse a single text record if present
        if not ndef:
            return None
        # NDEF header
        hdr = ndef[0]
        sr = (hdr & 0x10) != 0  # short record
        il = (hdr & 0x08) != 0  # id length present
        if 1 + 1 >= len(ndef):
            return None
        type_len = ndef[1]
        idx = 2
        if sr:
            if idx >= len(ndef):
                return None
            payload_len = ndef[idx]
            idx += 1
        else:
            if idx + 4 > len(ndef):
                return None
            payload_len = int.from_bytes(ndef[idx:idx+4], 'big')
            idx += 4
        if il:
            if idx >= len(ndef):
                return None
            id_len = ndef[idx]
            idx += 1 + id_len
        # Type
        rec_type = bytes(ndef[idx:idx+type_len])
        idx += type_len
        payload = bytes(ndef[idx:idx+payload_len])
        if rec_type == b'T' and payload:
            status = payload[0]
            lang_len = status & 0x3F
            is_utf16 = (status & 0x80) != 0
            text_bytes = payload[1+lang_len:]
            try:
                return text_bytes.decode('utf-16' if is_utf16 else 'utf-8', errors='ignore')
            except Exception:
                return text_bytes.decode('utf-8', errors='ignore')
        return None
    return None

@app.route('/rfid/read_record', methods=['GET'])
def rfid_read_record():
    return jsonify({'success': False, 'error': 'NDEF reading disabled'}), 410

@app.route('/rfid/read_block', methods=['POST'])
def rfid_read_block():
    try:
        body = request.get_json(silent=True) or {}
        block = int(body.get('block', 4))
        key_hex = (body.get('key_hex') or 'FFFFFFFFFFFF').strip()
        key_type = (body.get('key_type') or 'A').strip().upper()

        data, err = read_mifare_classic_block(block, key_hex=key_hex, key_type=key_type)
        if err:
            return jsonify({'success': False, 'error': err}), 400
        hex_str = data.hex().upper()
        try:
            text = data.decode('utf-8', errors='ignore')
        except Exception:
            text = ''
        return jsonify({'success': True, 'block': block, 'hex': hex_str, 'text': text})
    except Exception as e:
        app.logger.error(f"RFID read_block error: {e}")
        return jsonify({'success': False, 'error': 'Internal server error'}), 500

@app.route('/rfid/read_range', methods=['POST'])
def rfid_read_range():
    try:
        body = request.get_json(silent=True) or {}
        start_block = int(body.get('start_block', 4))
        num_blocks = int(body.get('num_blocks', 3))
        key_hex = (body.get('key_hex') or 'FFFFFFFFFFFF').strip()
        key_type = (body.get('key_type') or 'A').strip().upper()

        data, err = read_mifare_classic_range(start_block, num_blocks, key_hex=key_hex, key_type=key_type)
        if err:
            return jsonify({'success': False, 'error': err}), 400
        hex_str = data.hex().upper()
        try:
            text = data.decode('utf-8', errors='ignore')
        except Exception:
            text = ''
        return jsonify({'success': True, 'start_block': start_block, 'num_blocks': num_blocks, 'hex': hex_str, 'text': text})
    except Exception as e:
        app.logger.error(f"RFID read_range error: {e}")
        return jsonify({'success': False, 'error': 'Internal server error'}), 500

@app.route('/save_frame', methods=['POST'])
def save_frame():
    try:
        data = request.json
        image_data = data.get('frame')
        if not image_data:
            return jsonify({'error': 'No frame data received'}), 400

        detections = data.get('detections', [])
        image_bytes = base64.b64decode(image_data.split(',')[1])

        # Convert to cv2 image
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        conf_threshold = app.config['CONF_THRESHOLD']

        # Draw bounding boxes for high-confidence detections
        for det in detections:
            conf = float(det['confidence'])
            if conf >= conf_threshold:
                x1, y1, x2, y2 = map(int, det['bbox'])
                class_name = det['class']
                label = f"{class_name} {conf*100:.1f}%"

                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)

                text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                cv2.rectangle(img, (x1, y1 - 25), (x1 + text_size[0] + 10, y1), (0, 255, 0), -1)
                cv2.putText(img, label, (x1 + 5, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        filename = generate_filename()
        filepath = os.path.join(app.config['CAPTURE_FOLDER'], filename)
        cv2.imwrite(filepath, img)

        file_url = url_for('static', filename=f'captures/{filename}', _external=True)
        app.logger.info(f"Saved frame: {filename} with {len(detections)} detections.")

        return jsonify({'success': True, 'filepath': file_url})
    except Exception as e:
        app.logger.error(f"Error saving frame: {e}")
        return jsonify({'error': 'Internal server error while saving frame'}), 500

@app.route('/detect_frame', methods=['POST'])
def detect_frame():
    try:
        # Gate webcam detection strictly on live RFID presence
        _ensure_rfid_thread_running()
        if not _rfid_is_present():
            return jsonify({'error': 'RFID card must be present to detect'}), 403

        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400

        file = request.files['file']
        img_bytes = file.read()
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        results = model(img)
        result = results[0]
        detections = []

        conf_threshold = app.config['CONF_THRESHOLD']
        for box in result.boxes:
            conf = float(box.conf[0])
            if conf >= conf_threshold:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                class_id = int(box.cls[0])
                class_name = result.names[class_id]

                detections.append({
                    'bbox': [x1, y1, x2, y2],
                    'confidence': conf,
                    'class': class_name
                })

        return jsonify({'detections': detections})
    except Exception as e:
        app.logger.error(f"Error during frame detection: {e}")
        return jsonify({'error': 'Internal server error during detection'}), 500

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        if file and allowed_file(file.filename):
            filename = generate_filename(prefix='upload', extension=file.filename.rsplit('.', 1)[1])
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(filename))
            file.save(filepath)

            results = model(filepath)
            result = results[0]

            # Filter by confidence
            conf_threshold = app.config['CONF_THRESHOLD']
            result.boxes = result.boxes[result.boxes.conf >= conf_threshold]

            detections = []
            for box in result.boxes:
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(float, box.xyxy[0].tolist())
                class_id = int(box.cls[0])
                class_name = result.names[class_id]

                detections.append({
                    'bbox': [x1, y1, x2, y2],
                    'confidence': conf,
                    'class': class_name
                })

            annotated_img = result.plot()
            output_filename = f'result_{filename}'
            output_path = os.path.join('static', output_filename)
            cv2.imwrite(output_path, annotated_img)

            file_url = url_for('static', filename=output_filename, _external=True)
            app.logger.info(f"Prediction completed for {filename} with {len(detections)} detections.")

            return jsonify({'detections': detections, 'image_path': file_url})

        return jsonify({'error': 'Invalid file type'}), 400
    except Exception as e:
        app.logger.error(f"Prediction error: {e}")
        return jsonify({'error': 'Internal server error during prediction'}), 500

if __name__ == '__main__':
    app.run(debug=True)
