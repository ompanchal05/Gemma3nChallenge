from flask import Flask, render_template, Response, jsonify, redirect, url_for, request
import cv2
import pyttsx3
import requests
import numpy as np
import os
import sqlite3
from datetime import datetime
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import logging
import imutils

# -------------------- Logging Setup --------------------
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# -------------------- Flask App Setup --------------------
app = Flask(__name__)
API_KEY = os.getenv("GEMMA3N_API_KEY")

DB_FILE = "leaf_history.db"

# -------------------- Database Setup --------------------
# Import assistant functionality
from assistant import LeafAssistant

# Import optimized camera module
from camera import get_camera, initialize_camera, read_frame, release_camera

# Initialize assistant
assistant = LeafAssistant()
def get_predefined_response(leaf_name):
    predefined_responses = {
        "Holy Basil": {
            "info": "Holy Basil (Ocimum tenuiflorum), also known as Tulsi, is a sacred plant in Hinduism with many medicinal properties. It has anti-inflammatory, antioxidant, and immune-boosting effects. It's commonly used in Ayurveda for respiratory disorders, fever, and stress relief.",
            "advantages": [
                "Rich in antioxidants that help combat free radicals",
                "Boosts immunity and helps fight infections",
                "Reduces stress and anxiety",
                "Helps regulate blood sugar levels",
                "Supports respiratory health"
            ],
            "disadvantages": [
                "May lower blood sugar too much in diabetics",
                "Might slow blood clotting (caution before surgery)",
                "Possible interactions with blood-thinning medications",
                "Not recommended in large quantities during pregnancy"
            ],
            "medicinal_uses": [
                "Treating respiratory disorders like asthma and bronchitis",
                "Reducing fever and inflammation",
                "Managing diabetes and cholesterol levels",
                "Promoting heart health",
                "Enhancing skin health and treating skin disorders"
            ],
            "growing_tips": [
                "Prefers warm, tropical climate",
                "Needs well-drained soil and plenty of sunlight",
                "Water regularly but avoid waterlogging",
                "Prune regularly to encourage bushier growth",
                "Can be grown in pots or directly in the ground"
            ]
        },
        "Indian Lilac": {
            "info": "Indian Lilac (Azadirachta indica), commonly known as Neem, is a versatile medicinal tree native to India. It has antibacterial, antiviral, and antifungal properties.",
            "advantages": [
                "Powerful natural pesticide and insect repellent",
                "Promotes healthy skin and treats various skin conditions",
                "Supports oral health (used in toothpastes and mouthwashes)",
                "Helps purify blood in traditional medicine",
                "Has anti-diabetic properties"
            ],
            "disadvantages": [
                "Bitter taste makes it unpalatable for some uses",
                "May cause allergic reactions in sensitive individuals",
                "Can be toxic in large doses",
                "Not recommended for infants or pregnant women"
            ],
            "medicinal_uses": [
                "Treating skin conditions like eczema and acne",
                "Natural insect repellent and pesticide",
                "Oral care for healthy gums and teeth",
                "Blood purification in traditional medicine",
                "Managing diabetes and boosting immunity"
            ],
            "growing_tips": [
                "Thrives in hot, dry climates",
                "Drought-resistant once established",
                "Prefers well-drained soil",
                "Can grow up to 15-20 meters tall",
                "Fast-growing tree with many uses"
            ]
        },
        "Polyalthia Longifolia": {
            "info": "Polyalthia Longifolia, also known as the Mast Tree or False Ashoka, is an evergreen tree native to India. It's primarily ornamental but has some medicinal uses.",
            "advantages": [
                "Excellent noise pollution reducer (dense foliage)",
                "Used in traditional medicine for fever and skin diseases",
                "Provides good shade with minimal root damage",
                "Wind-resistant and low-maintenance tree",
                "Used in Ayurveda for its anti-inflammatory properties"
            ],
            "disadvantages": [
                "Limited medicinal uses compared to other plants",
                "Not as well-studied for therapeutic benefits",
                "Some people may be allergic to its pollen",
                "Large size makes it unsuitable for small gardens"
            ],
            "medicinal_uses": [
                "Fever reduction in traditional medicine",
                "Treating skin conditions and wounds",
                "Mild anti-inflammatory properties",
                "Sometimes used for digestive issues",
                "Occasional use in traditional pain relief"
            ],
            "growing_tips": [
                "Prefers tropical to subtropical climates",
                "Needs space to grow (can reach 30m tall)",
                "Low maintenance once established",
                "Tolerates various soil types",
                "Often planted as avenue trees for shade"
            ]
        }
    }
    return predefined_responses.get(leaf_name, None)

def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS detected_leaves (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            leaf_name TEXT,
            detected_time TEXT
        )
    """)
    conn.commit()
    conn.close()

init_db()

def log_leaf_detection(leaf_name):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("INSERT INTO detected_leaves (leaf_name, detected_time) VALUES (?, ?)", 
              (leaf_name, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    conn.commit()
    conn.close()

# -------------------- Text-to-Speech --------------------
try:
    engine = pyttsx3.init()
except Exception as e:
    logger.error(f"TTS Initialization Error: {e}")
    engine = None

def speak_text(text):
    if engine:
        try:
            engine.say(text)
            engine.runAndWait()
        except Exception as e:
            logger.error(f"TTS Error: {e}")

def speak_welcome():
    speak_text("Welcome Om Sir! AI-powered leaf detector is ready to use.")

# -------------------- Camera Setup --------------------
def init_camera():
    global video, greeted, current_leaf
    try:
        # Release existing camera if any
        if 'video' in globals() and video is not None:
            video.release()
        
        logger.info("🚀 Fast camera initialization starting...")
        
        # Use optimized camera module for fast opening
        camera = get_camera()
        if camera.initialize_camera():
            video = camera.cap
            # Reset greeting flag on new camera init
            greeted = False
            current_leaf = None
            logger.info("✅ Camera initialized successfully with optimized settings")
            return True
        else:
            logger.error("❌ Optimized camera initialization failed")
            return False
        
    except Exception as e:
        logger.error(f"❌ Camera Error: {e}")
        video = None
        return False

# Initialize camera at startup
video = None
greeted = False
current_leaf = None
init_camera()

# -------------------- Feature Extraction --------------------
def extract_features(img):
    if img is None or img.size == 0:
        return np.zeros(264)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [16, 16], [0, 180, 0, 256]).flatten()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        perimeter = cv2.arcLength(largest, True)
        x, y, w, h = cv2.boundingRect(largest)
        aspect_ratio = float(w) / h if h != 0 else 0
        circularity = 4 * np.pi * (area / (perimeter**2)) if perimeter != 0 else 0
        shape_features = np.array([area, perimeter, aspect_ratio, circularity])
    else:
        shape_features = np.zeros(4)

    cv2.normalize(hist, hist)
    return np.hstack([hist, shape_features])

# -------------------- Load Training Data --------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
expected_images = {
    "Holy Basil": ["holy_basil.jpg"],
    "Indian Lilac": ["indian_lilac.jpg"],
    "Polyalthia Longifolia": ["polyalthia_longifolia.jpg"]
}

X, y = [], []

for leaf_name, files in expected_images.items():
    for f in files:
        path = os.path.join(BASE_DIR, "static", "images", f)
        if os.path.exists(path):
            img = cv2.imread(path)
            if img is not None:
                X.append(extract_features(img))
                y.append(leaf_name)
                logger.info(f"Loaded: {path}")
            else:
                logger.warning(f"Failed to load image: {path}")
        else:
            logger.warning(f"Missing image: {path}")

if X:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    knn = KNeighborsClassifier(n_neighbors=min(3, len(set(y))))
    knn.fit(X_scaled, y)
    logger.info("✅ KNN Model Trained Successfully")
else:
    logger.error("❌ No Training Data Found!")
    scaler = None
    knn = None

# -------------------- Gemma3N API --------------------
def gemma3n_reply(prompt):
    if not API_KEY:
        return "API key is missing. Please set GEMMA3N_API_KEY in your environment."

    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
    headers = {"Content-Type": "application/json"}
    params = {"key": API_KEY}
    data = {"contents": [{"parts": [{"text": prompt}]}]}
    try:
        res = requests.post(url, headers=headers, params=params, json=data)
        return res.json()["candidates"][0]["content"]["parts"][0]["text"]
    except Exception as e:
        logger.error(f"API Error: {e}")
        return "Error fetching AI response."

# -------------------- Leaf Detection --------------------
def detect_leaf(frame):
    if knn is None or scaler is None:
        return None, None

    try:
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Broader green color range for better leaf detection
        # Lower bound: darker greens, Upper bound: lighter greens
        lower_green = np.array([25, 20, 20])  # More permissive lower bound
        upper_green = np.array([95, 255, 255])  # Keep upper bound
        
        # Create mask for green objects
        mask = cv2.inRange(hsv, lower_green, upper_green)

        # Apply morphological operations to clean up the mask
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)

        if cnts:
            # Sort contours by area and get the largest ones
            cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
            
            for contour in cnts[:3]:  # Check top 3 largest contours
                area = cv2.contourArea(contour)
                
                # Lower threshold for better detection
                if area > 500:  # Reduced from 2000 to 500
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # Skip if bounding box is too small
                    if w < 20 or h < 20:
                        continue
                    
                    # Extract ROI (Region of Interest)
                    roi = frame[y:y+h, x:x+w]
                    
                    # Skip if ROI is invalid
                    if roi.size == 0:
                        continue
                    
                    # Extract features and predict
                    features = extract_features(roi)
                    if features is not None:
                        try:
                            pred = knn.predict(scaler.transform([features]))[0]
                            return pred, (x, y, w, h)
                        except Exception as e:
                            logger.error(f"Prediction error: {e}")
                            continue
        
        return None, None
    except Exception as e:
        logger.error(f"Detection Error: {e}")
        return None, None

# -------------------- Global Variables for Leaf Detection --------------------
detected_leaf_signal = None
last_detected_leaf = None

# -------------------- Frame Generator --------------------
def generate_frames():
    global greeted, current_leaf, detected_leaf_signal, last_detected_leaf, video

    # Ensure camera is initialized
    if not video or not video.isOpened():
        logger.info("🔄 Reinitializing camera for video stream...")
        if not init_camera():
            logger.error("❌ Failed to initialize camera for video stream")
            return

    logger.info("🎥 Starting video stream generation")
    
    while True:
        try:
            # Check if camera is still working
            if not video or not video.isOpened():
                logger.error("❌ Camera connection lost, attempting to reconnect...")
                if not init_camera():
                    logger.error("❌ Failed to reconnect camera")
                    break
                continue

            # Read frame directly from video object to avoid black screen
            success, frame = video.read()
            if not success or frame is None:
                logger.error("❌ Failed to read frame from camera")
                # Try to reinitialize camera
                if not init_camera():
                    break
                continue

            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)

            if not greeted:
                try:
                    speak_welcome()
                    greeted = True
                except Exception:
                    pass

            # Add status information to frame
            status_text = "Status: Ready"
            cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Add model status
            model_status = "Model: " + ("Ready" if knn is not None else "Not Ready")
            cv2.putText(frame, model_status, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if knn is not None else (0, 0, 255), 2)

            # Detect leaf
            leaf, box = detect_leaf(frame)
            
            if leaf and box:
                x, y, w, h = box
                # Draw green rectangle around detected leaf
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)
                cv2.putText(frame, f"{leaf} Detected", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 3)
                
                # Update status
                cv2.putText(frame, f"Status: {leaf} Detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                if current_leaf != leaf:
                    current_leaf = leaf
                    last_detected_leaf = leaf
                    detected_leaf_signal = leaf  # Signal for frontend
                    
                    try:
                        speak_text(f"{leaf} detected.")
                        log_leaf_detection(leaf)
                        
                        # Also speak with assistant if available
                        try:
                            assistant.speak(f"I detected {leaf}! This is an interesting plant with many properties.")
                        except:
                            pass  # Ignore if assistant speech fails
                    except:
                        pass
            else:
                # Show "No leaf detected" message
                cv2.putText(frame, "No leaf detected", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame, "Status: Searching for leaves...", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

            # Add instructions
            cv2.putText(frame, "Hold a leaf in front of the camera", (10, frame.shape[0] - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, "Make sure the leaf is well-lit and clearly visible", (10, frame.shape[0] - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Encode frame to JPEG
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if not ret:
                logger.error("❌ Failed to encode frame")
                continue
                
            frame_bytes = buffer.tobytes()
            
            # Yield frame in MIME format
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n'
                   b'Content-Length: ' + str(len(frame_bytes)).encode() + b'\r\n'
                   b'\r\n' + frame_bytes + b'\r\n')
        except Exception as e:
            logger.error(f"❌ Error in generate_frames: {e}")
            # Try to reinitialize camera on error
            try:
                if video:
                    video.release()
                video = None
                if not init_camera():
                    break
            except:
                break

            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)

            if not greeted:
                try:
                    speak_welcome()
                    greeted = True
                except Exception:
                    pass

            # Add status information to frame
            status_text = "Status: Ready"
            cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Add model status
            model_status = "Model: " + ("Ready" if knn is not None else "Not Ready")
            cv2.putText(frame, model_status, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if knn is not None else (0, 0, 255), 2)

            # Detect leaf
            leaf, box = detect_leaf(frame)
            
            if leaf and box:
                x, y, w, h = box
                # Draw green rectangle around detected leaf
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)
                cv2.putText(frame, f"{leaf} Detected", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 3)
                
                # Update status
                cv2.putText(frame, f"Status: {leaf} Detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                if current_leaf != leaf:
                    current_leaf = leaf
                    last_detected_leaf = leaf
                    detected_leaf_signal = leaf  # Signal for frontend
                    
                    try:
                        speak_text(f"{leaf} detected.")
                        log_leaf_detection(leaf)
                        
                        # Also speak with assistant if available
                        try:
                            assistant.speak(f"I detected {leaf}! This is an interesting plant with many properties.")
                        except:
                            pass  # Ignore if assistant speech fails
                    except:
                        pass
            else:
                # Show "No leaf detected" message
                cv2.putText(frame, "No leaf detected", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame, "Status: Searching for leaves...", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

            # Add instructions
            cv2.putText(frame, "Hold a leaf in front of the camera", (10, frame.shape[0] - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, "Make sure the leaf is well-lit and clearly visible", (10, frame.shape[0] - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Encode frame to JPEG
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if not ret:
                logger.error("❌ Failed to encode frame")
                continue
                
            frame_bytes = buffer.tobytes()
            
            # Yield frame in MIME format
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n'
                   b'Content-Length: ' + str(len(frame_bytes)).encode() + b'\r\n'
                   b'\r\n' + frame_bytes + b'\r\n')
        except Exception as e:
            logger.error(f"❌ Error in generate_frames: {e}")
            # Try to reinitialize camera on error
            try:
                if video:
                    video.release()
                video = None
                if not init_camera():
                    break
            except:
                break
    
    logger.info("🎥 Video stream generation ended")

# -------------------- Routes --------------------
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/leaf_info/<leaf>')
def leaf_info(leaf):
    predefined = get_predefined_response(leaf)
    if predefined:
        # If we have predefined data, use it
        return render_template("leaf_info.html", 
                             leaf=leaf, 
                             info=predefined["info"],
                             advantages=predefined["advantages"],
                             disadvantages=predefined["disadvantages"],
                             medicinal_uses=predefined["medicinal_uses"],
                             growing_tips=predefined["growing_tips"])
    else:
        # Fallback to API for unknown leaves
        info = gemma3n_reply(f"Explain advantages and disadvantages of {leaf}.")
        return render_template("leaf_info.html", leaf=leaf, info=info)

@app.route('/restart_camera')
def restart_camera():
    if init_camera():
        return jsonify({"status": "success", "message": "Camera reinitialized"})
    return jsonify({"status": "error", "message": "Failed to reinitialize camera"}), 500

@app.route('/camera_status')
def camera_status():
    """Check camera status"""
    try:
        if video is None:
            return jsonify({"status": "error", "message": "Camera not initialized"})
        
        if not video.isOpened():
            return jsonify({"status": "error", "message": "Camera not opened"})
        
        # Try to read a test frame
        ret, frame = video.read()
        if not ret:
            return jsonify({"status": "error", "message": "Cannot read frames from camera"})
        
        return jsonify({
            "status": "success", 
            "message": "Camera working",
            "frame_shape": frame.shape,
            "camera_opened": video.isOpened()
        })
        
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/camera_debug')
def camera_debug():
    """Debug route to show detailed camera information"""
    debug_info = {
        "video_object": "None" if video is None else "Initialized",
        "camera_opened": video.isOpened() if video else False,
        "camera_backend": "Unknown"
    }
    
    if video:
        try:
            # Try to get camera properties
            width = video.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = video.get(cv2.CAP_PROP_FRAME_HEIGHT)
            fps = video.get(cv2.CAP_PROP_FPS)
            
            debug_info.update({
                "frame_width": width,
                "frame_height": height,
                "fps": fps
            })
            
            # Try to read a test frame
            ret, frame = video.read()
            if ret:
                debug_info["test_frame_shape"] = frame.shape
                debug_info["test_frame_success"] = True
            else:
                debug_info["test_frame_success"] = False
                
        except Exception as e:
            debug_info["error"] = str(e)
    
    return jsonify(debug_info)

def get_detected_leaves():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT leaf_name, detected_time FROM detected_leaves ORDER BY id DESC LIMIT 20")
    rows = c.fetchall()
    conn.close()
    return rows

@app.route('/detected_leaves')
def detected_leaves():
    return render_template("detected_leaves.html", leaves=get_detected_leaves())

@app.route('/about')
def about():
    return render_template("about.html")

@app.route('/contact')
def contact():
    return render_template("contact.html")

@app.route('/assistant')
def assistant_page():
    return render_template("assistant.html")

@app.route('/assistant/speak/<leaf>')
def assistant_speak(leaf):
    """API endpoint to make assistant speak about a leaf"""
    try:
        if leaf.lower() in ['holy basil', 'tulsi', 'basil']:
            assistant.speak_leaf_info("Holy Basil")
        elif leaf.lower() in ['indian lilac', 'neem', 'lilac']:
            assistant.speak_leaf_info("Indian Lilac")
        elif leaf.lower() in ['polyalthia', 'mast tree', 'polyalthia longifolia']:
            assistant.speak_leaf_info("Polyalthia Longifolia")
        elif leaf.lower() in ['all', 'summary']:
            assistant.speak_all_leaves_summary()
        else:
            assistant.speak(f"Sorry, I don't have information about {leaf}")
        
        return jsonify({"status": "success", "message": f"Spoke about {leaf}"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/assistant/speak/text', methods=['POST'])
def assistant_speak_text():
    """API endpoint to make assistant speak custom text"""
    try:
        data = request.get_json()
        text = data.get('text', 'Hello!')
        
        # Use the assistant to speak
        assistant.speak(text)
        
        return jsonify({"status": "success", "message": "Text spoken successfully"})
    except Exception as e:
        logger.error(f"Text-to-speech error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/debug')
def debug_info():
    """Debug route to show system status"""
    debug_info = {
        "camera_status": "Connected" if video and video.isOpened() else "Not Connected",
        "model_status": "Ready" if knn is not None else "Not Ready",
        "scaler_status": "Ready" if scaler is not None else "Not Ready",
        "training_images": [],
        "training_data_count": len(X) if 'X' in globals() else 0,
        "unique_classes": len(set(y)) if 'y' in globals() else 0
    }
    
    # Check training images
    for leaf_name, files in expected_images.items():
        for f in files:
            path = os.path.join(BASE_DIR, "static", "images", f)
            if os.path.exists(path):
                size = os.path.getsize(path)
                debug_info["training_images"].append({
                    "name": f,
                    "exists": True,
                    "size_bytes": size,
                    "size_kb": round(size / 1024, 2)
                })
            else:
                debug_info["training_images"].append({
                    "name": f,
                    "exists": False,
                    "size_bytes": 0,
                    "size_kb": 0
                })
    
    return render_template("debug.html", debug_info=debug_info)

@app.route('/create_training_data')
def create_training_data_page():
    """Page to create training data"""
    return render_template("create_training_data.html")

@app.route('/capture_image/<leaf_type>')
def capture_image(leaf_type):
    """Capture training image for a specific leaf type"""
    try:
        if not video or not video.isOpened():
            return jsonify({"status": "error", "message": "Camera not available"}), 500
        
        # Capture frame
        success, frame = video.read()
        if not success:
            return jsonify({"status": "error", "message": "Failed to capture frame"}), 500
        
        # Flip frame
        frame = cv2.flip(frame, 1)
        
        # Save image
        filename = f"{leaf_type}.jpg"
        filepath = os.path.join(BASE_DIR, "static", "images", filename)
        cv2.imwrite(filepath, frame)
        
        # Retrain model
        retrain_model()
        
        return jsonify({
            "status": "success", 
            "message": f"Captured {leaf_type} image successfully",
            "filename": filename
        })
        
    except Exception as e:
        logger.error(f"Capture error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

def retrain_model():
    """Retrain the ML model with new data"""
    global X, y, scaler, knn
    
    X, y = [], []
    
    for leaf_name, files in expected_images.items():
        for f in files:
            path = os.path.join(BASE_DIR, "static", "images", f)
            if os.path.exists(path):
                img = cv2.imread(path)
                if img is not None and img.size > 0:
                    X.append(extract_features(img))
                    y.append(leaf_name)
                    logger.info(f"Retrained with: {path}")
    
    if X:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        knn = KNeighborsClassifier(n_neighbors=min(3, len(set(y))))
        knn.fit(X_scaled, y)
        logger.info("✅ Model retrained successfully")
    else:
        logger.error("❌ No training data available for retraining")
        scaler = None
        knn = None

@app.route('/create_dummy_data')
def create_dummy_data():
    """Create dummy training images for testing"""
    try:
        # Create simple colored rectangles as dummy images
        leaves = {
            'holy_basil': (0, 100, 0),      # Dark green
            'indian_lilac': (0, 150, 0),    # Medium green  
            'polyalthia_longifolia': (0, 200, 0)  # Light green
        }
        
        for leaf_name, color in leaves.items():
            # Create a 300x300 image with the specified color
            img = np.full((300, 300, 3), color, dtype=np.uint8)
            
            # Add some variation to make it look more like a leaf
            # Add some darker spots
            for _ in range(10):
                x = np.random.randint(50, 250)
                y = np.random.randint(50, 250)
                cv2.circle(img, (x, y), np.random.randint(5, 15), (0, 50, 0), -1)
            
            # Add some lighter spots
            for _ in range(5):
                x = np.random.randint(50, 250)
                y = np.random.randint(50, 250)
                cv2.circle(img, (x, y), np.random.randint(3, 10), (0, 255, 0), -1)
            
            filename = f"{leaf_name}.jpg"
            filepath = os.path.join(BASE_DIR, "static", "images", filename)
            cv2.imwrite(filepath, img)
            logger.info(f"Created dummy image: {filename}")
        
        # Retrain model with new data
        retrain_model()
        
        return jsonify({
            "status": "success",
            "message": "Dummy training data created successfully"
        })
        
    except Exception as e:
        logger.error(f"Dummy data creation error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/camera_test')
def camera_test():
    """Camera test page"""
    return render_template("camera_test.html")

@app.route('/check_detected_leaf')
def check_detected_leaf():
    """Check if a new leaf was detected and return it"""
    global detected_leaf_signal
    
    if detected_leaf_signal:
        leaf = detected_leaf_signal
        detected_leaf_signal = None  # Clear the signal
        return jsonify({
            "status": "success",
            "detected": True,
            "leaf": leaf
        })
    
    return jsonify({
        "status": "success",
        "detected": False,
        "leaf": None
    })

@app.route('/test_camera_frame')
def test_camera_frame():
    """Test route to capture a single frame and return it as an image"""
    try:
        # Ensure camera is initialized
        if not video or not video.isOpened():
            if not init_camera():
                return "Camera not available", 500
        
        # Capture a single frame
        success, frame = video.read()
        if not success:
            return "Failed to capture frame", 500
        
        # Flip frame
        frame = cv2.flip(frame, 1)
        
        # Add test text
        cv2.putText(frame, "Camera Test Frame", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, "If you can see this, camera is working!", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Encode to JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            return "Failed to encode frame", 500
        
        # Return as image
        from flask import send_file
        import io
        return send_file(
            io.BytesIO(buffer.tobytes()),
            mimetype='image/jpeg'
        )
        
    except Exception as e:
        logger.error(f"Test camera frame error: {e}")
        return f"Error: {str(e)}", 500

# -------------------- Main --------------------
if __name__ == "__main__":
    app.run(debug=False, host='127.0.0.1', port=5000)
