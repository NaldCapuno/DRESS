from flask import Flask, render_template, request, jsonify, url_for, Response, session, redirect
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
from database import db_manager

# ------------------ Flask App Setup ------------------
app = Flask(__name__)

# ------------------ Configuration ------------------
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['CAPTURE_FOLDER'] = 'static/captures'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['CONF_THRESHOLD'] = 0.5  # Lowered from 0.85 for debugging
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Session secret key
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'dev-insecure-change-me')

# Required attire per gender
REQUIRED_UNIFORM_BY_GENDER = {
    'male': {'polo_shirt', 'pants', 'shoes'},
    'female': {'blouse', 'skirt', 'doll_shoes'},
}

# ------------------ Capture Throttling ------------------
# Capture only every N frames and up to M images per RFID session
app.config['CAPTURE_EVERY_N_FRAMES'] = int(os.getenv('CAPTURE_EVERY_N_FRAMES', '2'))  # Faster: every 2nd frame
app.config['CAPTURE_MAX_IMAGES'] = int(os.getenv('CAPTURE_MAX_IMAGES', '3'))  # Only 3 total images

_capture_session = {
    'uid': None,
    'frame_index': 0,
    'saved_count': 0,
    'start_time': 0.0,
}

def _reset_capture_session(uid: str | None):
    _capture_session['uid'] = uid
    _capture_session['frame_index'] = 0
    _capture_session['saved_count'] = 0
    _capture_session['start_time'] = time.time()

def _should_capture_this_frame(current_uid: str | None) -> tuple[bool, int, int]:
    # Returns (should_capture, frame_index, saved_count)
    if current_uid != _capture_session.get('uid'):
        _reset_capture_session(current_uid)

    _capture_session['frame_index'] += 1
    n = app.config['CAPTURE_EVERY_N_FRAMES']
    max_imgs = app.config['CAPTURE_MAX_IMAGES']

    if _capture_session['saved_count'] >= max_imgs:
        return False, _capture_session['frame_index'], _capture_session['saved_count']

    if (_capture_session['frame_index'] % max(1, n)) == 0:
        return True, _capture_session['frame_index'], _capture_session['saved_count']

    return False, _capture_session['frame_index'], _capture_session['saved_count']

# Violation throttling (per RFID session)
app.config['VIOLATION_MAX_IMAGES'] = int(os.getenv('VIOLATION_MAX_IMAGES', '3'))  # Faster: only 3 violation captures needed

_violation_session = {
    'uid': None,
    'created_count': 0,
    'capture_count': 0,  # Track captures for violations
    'start_time': 0.0,
}

def _reset_violation_session(uid: str | None):
    _violation_session['uid'] = uid
    _violation_session['created_count'] = 0
    _violation_session['capture_count'] = 0
    _violation_session['start_time'] = time.time()

def _can_create_violation_for_uid(uid: str | None) -> bool:
    if uid != _violation_session.get('uid'):
        _reset_violation_session(uid)
    return _violation_session['created_count'] < app.config['VIOLATION_MAX_IMAGES']

def _reset_all_sessions():
    """Reset both capture and violation sessions"""
    _reset_capture_session(None)
    _reset_violation_session(None)

def _delete_first_captured_images():
    """Delete the first 5 captured images for the current session"""
    try:
        capture_folder = os.path.join(app.root_path, 'static', 'captures')
        if not os.path.exists(capture_folder):
            return
        
        # Get only regular captured images (not violation images), sorted by modification time
        image_files = []
        for filename in os.listdir(capture_folder):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')) and filename.startswith('img_'):
                filepath = os.path.join(capture_folder, filename)
                if os.path.isfile(filepath):
                    image_files.append((filepath, os.path.getmtime(filepath)))
        
        # Sort by modification time (oldest first)
        image_files.sort(key=lambda x: x[1])
        
        # Delete the first 3 regular images (oldest)
        deleted_count = 0
        for filepath, _ in image_files[:3]:
            try:
                os.remove(filepath)
                deleted_count += 1
                app.logger.info(f"Deleted captured image: {os.path.basename(filepath)}")
            except Exception as e:
                app.logger.error(f"Failed to delete {filepath}: {e}")
        
        app.logger.info(f"Deleted {deleted_count} captured images after violation creation")
        
    except Exception as e:
        app.logger.error(f"Error deleting captured images: {e}")



def _filter_detections_by_gender(detections: list[dict], gender: str | None) -> list[dict]:
    """Filter detections to only include items relevant to the student's gender."""
    if not gender:
        app.logger.info("No gender provided, returning all detections")
        return detections
    
    gender_key = str(gender).strip().lower()
    required_items = REQUIRED_UNIFORM_BY_GENDER.get(gender_key, set())
    
    if not required_items:
        app.logger.info(f"No required items found for gender {gender_key}, returning all detections")
        return detections
    
    # Filter detections to only include items relevant to this gender
    filtered_detections = []
    for detection in detections:
        class_name = str(detection.get('class', '')).strip().lower()
        
        # Include the detection if it's a required item for this gender
        if class_name in required_items:
            filtered_detections.append(detection)
            app.logger.info(f"Including detection: {class_name} (required for {gender_key})")
        else:
            app.logger.info(f"Filtering out detection: {class_name} (not required for {gender_key})")
    
    app.logger.info(f"Filtered {len(detections)} detections to {len(filtered_detections)} gender-relevant detections")
    return filtered_detections


def _evaluate_uniform_completeness(detections: list[dict], gender: str | None) -> tuple[set[str], set[str]]:
    # returns (present_items, missing_items)
    detected_classes = {str(d.get('class', '')).strip().lower() for d in detections}
    app.logger.info(f"Uniform evaluation - gender: {gender}, detected_classes: {detected_classes}")
    
    if not gender:
        app.logger.info("No gender provided, returning detected classes only")
        return detected_classes, set()  # Without gender we cannot decide
    
    gender_key = str(gender).strip().lower()
    required = REQUIRED_UNIFORM_BY_GENDER.get(gender_key)
    app.logger.info(f"Required items for {gender_key}: {required}")
    
    if not required:
        app.logger.info("No required items found for gender")
        return detected_classes, set()
    
    missing = {item for item in required if item not in detected_classes}
    app.logger.info(f"Missing items: {missing}")
    return detected_classes, missing

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
                match_info = db_manager.lookup_and_log(uid)
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
    return render_template('index.html')

@app.route('/login')
def login():
    return render_template('login.html')

@app.route('/login', methods=['POST'])
def login_post():
    try:
        data = request.get_json(silent=True) or {}
        # Support both keys for backward compatibility
        username = (data.get('username') or data.get('email') or '').strip()
        password = (data.get('password') or '').strip()

        if not username or not password:
            return jsonify({'success': False, 'error': 'Please fill in all fields'}), 400

        admin = db_manager.verify_admin_credentials(username, password)
        if not admin:
            return jsonify({'success': False, 'error': 'Invalid username or password'}), 401

        # Store minimal admin info in session
        session['admin'] = {
            'admin_id': admin.get('admin_id'),
            'username': admin.get('username'),
            'role': (admin.get('role') or '').lower()
        }

        return jsonify({'success': True, 'message': 'Login successful', 'admin': session['admin']})
    except Exception as e:
        app.logger.error(f"Login error: {e}")
        return jsonify({'success': False, 'error': 'Login failed. Please try again.'}), 500

@app.route('/dashboard')
def dashboard():
    admin = session.get('admin')
    if not admin:
        return redirect(url_for('login'))
    if (admin.get('role') or '').lower() != 'security':
        return jsonify({'success': False, 'error': 'Forbidden: security role required'}), 403
    return render_template('dashboard.html')

# ------------------ OSAS Dashboard ------------------
def _require_role(role_required: str):
    admin = session.get('admin')
    if not admin:
        return None, (redirect(url_for('login')))
    if (admin.get('role') or '').lower() != role_required.lower():
        return None, (jsonify({'success': False, 'error': 'Forbidden'}), 403)
    return admin, None

@app.route('/osas')
def osas_dashboard():
    admin, err = _require_role('osas')
    if err:
        return err
    return render_template('osas_dashboard.html')

@app.route('/osas/violations', methods=['GET'])
def osas_violations():
    admin, err = _require_role('osas')
    if err:
        return err
    # Filters
    start_dt = request.args.get('start')
    end_dt = request.args.get('end')
    ay = request.args.get('academic_year')
    sem = request.args.get('semester')
    page = int(request.args.get('page', '1'))
    page_size = int(request.args.get('page_size', '50'))
    offset = (max(page, 1) - 1) * page_size
    data = db_manager.get_violations(start_dt=start_dt, end_dt=end_dt, academic_year=ay, semester=sem, limit=page_size, offset=offset)
    return jsonify({'success': True, **data})

@app.route('/osas/analytics', methods=['GET'])
def osas_analytics():
    admin, err = _require_role('osas')
    if err:
        return err
    start_dt = request.args.get('start')
    end_dt = request.args.get('end')
    ay = request.args.get('academic_year')
    sem = request.args.get('semester')
    data = db_manager.get_violations_analytics(start_dt=start_dt, end_dt=end_dt, academic_year=ay, semester=sem)
    return jsonify({'success': True, **data})

@app.route('/osas/trend', methods=['GET'])
def osas_trend():
    admin, err = _require_role('osas')
    if err:
        return err
    start_dt = request.args.get('start')
    end_dt = request.args.get('end')
    ay = request.args.get('academic_year')
    sem = request.args.get('semester')
    group_by = request.args.get('group_by', 'day')
    data = db_manager.get_violations_trend(start_dt=start_dt, end_dt=end_dt, academic_year=ay, semester=sem, group_by=group_by)
    return jsonify({'success': True, **data})

@app.route('/osas/violation/<int:violation_id>/status', methods=['POST'])
def osas_update_status(violation_id: int):
    admin, err = _require_role('osas')
    if err:
        return err
    body = request.get_json(silent=True) or {}
    status = (body.get('status') or '').strip().lower()
    # Map requested workflow names to DB statuses
    # Forwarded to Dean / Guidance are both represented as 'forwarded'
    status_map = {
        'forwarded_to_dean': 'forwarded',
        'forwarded_to_guidance': 'forwarded',
        'resolved': 'resolved',
        'pending': 'pending'
    }
    mapped = status_map.get(status, status)
    ok = db_manager.update_violation_status(violation_id, mapped)
    if not ok:
        return jsonify({'success': False, 'error': 'Invalid status or update failed'}), 400
    return jsonify({'success': True})

@app.route('/logout', methods=['POST'])
def logout():
    session.pop('admin', None)
    return jsonify({'success': True, 'message': 'Logged out'})

@app.route('/rfid/read_uid', methods=['GET'])
def rfid_read_uid():
    try:
        _ensure_rfid_thread_running()
        uid, err = get_rfid_uid()
        if uid:
            _rfid_set_present(True)
            match_info = db_manager.lookup_and_log(uid)
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
        student = db_manager.find_student_by_rfid(uid)
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

@app.route('/reset_sessions', methods=['POST'])
def reset_sessions():
    try:
        _reset_all_sessions()
        app.logger.info("Reset capture and violation sessions")
        return jsonify({'success': True, 'message': 'Sessions reset successfully'})
    except Exception as e:
        app.logger.error(f"Session reset error: {e}")
        return jsonify({'success': False, 'error': 'Internal server error'}), 500

@app.route('/save_frame', methods=['POST'])
def save_frame():
    try:
        data = request.json
        image_data = data.get('frame')
        if not image_data:
            return jsonify({'error': 'No frame data received'}), 400

        detections = data.get('detections', [])
        # Get student information and filter detections by gender
        gender = None
        student = None
        try:
            uid = _rfid_last_uid
            if uid:
                student = db_manager.find_student_by_rfid(uid)
                if student:
                    gender = student.get('gender')
        except Exception:
            pass
        
        # Filter detections to only include items relevant to the student's gender
        filtered_detections = _filter_detections_by_gender(detections, gender)
        present_items, missing_items = _evaluate_uniform_completeness(filtered_detections, gender)

        image_bytes = base64.b64decode(image_data.split(',')[1])

        # Convert to cv2 image
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        conf_threshold = app.config['CONF_THRESHOLD']

        # Decide whether to capture this frame (throttled per RFID uid)
        current_uid = _rfid_last_uid
        should_capture, frame_index, saved_count = _should_capture_this_frame(current_uid)

        if not should_capture:
            is_final_slot = saved_count >= (app.config['CAPTURE_MAX_IMAGES'] - 1)
            return jsonify({'success': True, 'captured': False, 'reason': 'throttled', 'frame_index': frame_index, 'saved_count': saved_count, 'final_slot': is_final_slot})

        # Draw bounding boxes for high-confidence filtered detections
        for det in filtered_detections:
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

        _capture_session['saved_count'] += 1

        file_url = url_for('static', filename=f'captures/{filename}', _external=True)
        app.logger.info(f"Saved frame: {filename} with {len(detections)} detections. saved_count={_capture_session['saved_count']}")

        # Track violation captures separately and only insert after 5th violation capture
        violation_created = False
        violation_id = None
        violation_image_url = None
        
        # Debug logging
        app.logger.info(f"Save frame - student: {student is not None}, gender: {gender}, missing_items: {missing_items}, total_detections: {len(detections)}, filtered_detections: {len(filtered_detections)}")
        app.logger.info(f"Session - saved_count: {_capture_session['saved_count']}, violation_created: {_violation_session['created_count']}, uid: {_rfid_last_uid}")
        
        # Ensure violation session is set up for this UID
        if _rfid_last_uid != _violation_session.get('uid'):
            _reset_violation_session(_rfid_last_uid)
            app.logger.info(f"Reset violation session for new UID: {_rfid_last_uid}")
        
        # Check if this frame has a violation (missing items) and create violation after 3rd regular capture
        if student and gender and missing_items and _capture_session['saved_count'] == 3 and _can_create_violation_for_uid(_rfid_last_uid):
            annotated = img.copy()
            missing_text = "Missing: " + ", ".join(sorted(missing_items))
            cv2.putText(annotated, missing_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)
            for det in filtered_detections:
                conf = float(det['confidence'])
                if conf >= conf_threshold:
                    x1, y1, x2, y2 = map(int, det['bbox'])
                    label = f"{det['class']} {conf*100:.1f}%"
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 3)
                    text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                    cv2.rectangle(annotated, (x1, y1 - 25), (x1 + text_size[0] + 10, y1), (0, 0, 255), -1)
                    cv2.putText(annotated, label, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            v_filename = generate_filename(prefix='violation', extension='jpg')
            v_rel_path = os.path.join('static', 'captures', v_filename)
            v_abs_path = os.path.join(app.root_path, 'static', 'captures', v_filename)
            os.makedirs(os.path.dirname(v_abs_path), exist_ok=True)
            cv2.imwrite(v_abs_path, annotated)
            violation_image_url = url_for('static', filename=f'captures/{v_filename}', _external=True)

            violation_desc = f"Incomplete uniform for {gender}: missing {', '.join(sorted(missing_items))}"
            violation_id = db_manager.insert_violation(student.get('student_id'), violation_desc, v_rel_path, recorded_by=None)
            violation_created = violation_id is not None
            if violation_created:
                _violation_session['created_count'] += 1
                
                # Delete the first 3 captured images for this session
                _delete_first_captured_images()

        return jsonify({
            'success': True, 
            'captured': True, 
            'filepath': file_url, 
            'frame_index': frame_index, 
            'saved_count': _capture_session['saved_count'], 
            'violation': {
                'created': violation_created, 
                'violation_id': violation_id, 
                'image_url': violation_image_url, 
                'created_count': _violation_session['created_count'] if _rfid_last_uid == _violation_session.get('uid') else 0,
                'capture_count': _violation_session['capture_count'] if _rfid_last_uid == _violation_session.get('uid') else 0,
                'limit': app.config.get('VIOLATION_MAX_IMAGES')
            }
        })
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

        # Require that the last scanned RFID corresponds to a registered student
        uid = _rfid_last_uid
        if not uid:
            return jsonify({'error': 'No RFID uid captured'}), 403
        student_check = db_manager.find_student_by_rfid(uid)
        if not student_check:
            return jsonify({'error': 'RFID is not registered to any student'}), 403

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

        # Get student information and filter detections by gender
        student = None
        gender = None
        try:
            uid = _rfid_last_uid
            if uid:
                student = db_manager.find_student_by_rfid(uid)
                if student:
                    gender = student.get('gender')
        except Exception:
            pass

        # Filter detections to only include items relevant to the student's gender
        filtered_detections = _filter_detections_by_gender(detections, gender)
        
        # Evaluate uniform completeness based on filtered detections
        present_items, missing_items = _evaluate_uniform_completeness(filtered_detections, gender)
        
        # Debug logging
        app.logger.info(f"Detect frame - student: {student is not None}, gender: {gender}, total_detections: {len(detections)}, filtered_detections: {len(filtered_detections)}, present: {present_items}, missing: {missing_items}")

        # Defer violation creation to end-of-capture in save_frame
        violation_created = False
        violation_id = None
        image_url = None

        response = {
            'detections': filtered_detections,  # Use filtered detections instead of all detections
            'uniform': {
                'gender': gender,
                'present': sorted(list(present_items)) if present_items else [],
                'missing': sorted(list(missing_items)) if missing_items else []
            },
            'violation': {
                'created': False,
                'note': 'violation insertion deferred to end-of-capture if still missing',
                'created_count': _violation_session['created_count'] if _rfid_last_uid == _violation_session.get('uid') else 0,
                'limit': app.config.get('VIOLATION_MAX_IMAGES')
            }
        }

        return jsonify(response)
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
