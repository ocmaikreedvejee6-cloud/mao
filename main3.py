import cv2
import numpy as np
import serial
import time
import requests
import smtplib
import os
import threading
from datetime import datetime, timedelta, timezone
from email.message import EmailMessage
from flask import Flask, Response

# ================= CONFIG =================
BAUD_RATE = 9600

TELEGRAM_TOKEN = "8490765768:AAFU-Vpi0HAiS5_2V2mcboWYeiG8W4neiVE"
CHAT_ID = "7175315173"

CONFIDENCE_THRESHOLD = 50
FACE_TIMEOUT = 3
TELEGRAM_COOLDOWN = 30

EMAIL_ADDRESS = "growpfiveim312@gmail.com"
EMAIL_PASSWORD = "qerlwnbhfcaprcll"
RECEIVER_EMAIL = "ocmaikreedvejee6@gmail.com"

PH_TIMEZONE = timezone(timedelta(hours=8))

# ================= STATES =================
last_face_time = 0
last_telegram_time = 0
unknown_triggered = False
relay_state = False

arduino = None
cap = None
frame_global = None
lock = threading.Lock()

# ================= AUTO ARDUINO CONNECT =================
def connect_arduino():
    global arduino

    ports = ["/dev/ttyACM0", "/dev/ttyACM1", "/dev/ttyUSB0", "/dev/ttyUSB1"]

    for p in ports:
        try:
            arduino = serial.Serial(p, BAUD_RATE, timeout=1)
            time.sleep(2)
            print(f"✅ Arduino connected: {p}")
            return True
        except:
            continue

    arduino = None
    print("❌ Arduino not found")
    return False

def safe_write(cmd):
    global arduino

    if arduino is None:
        connect_arduino()
        return

    try:
        arduino.write(cmd.encode())
    except:
        print("⚠️ Arduino lost, reconnecting...")
        arduino = None
        connect_arduino()

# ================= AUTO CAMERA DETECT =================
def find_camera():
    global cap

    for i in range(5):
        temp = cv2.VideoCapture(i)
        if temp.isOpened():
            print(f"✅ Camera found at index {i}")
            cap = temp
            return True
        temp.release()

    print("❌ No camera found")
    return False

# ================= RELAY CONTROL =================
def set_relays(state):
    global relay_state

    if relay_state == state:
        return

    relay_state = state

    if state:
        safe_write("ON\n")
        print("💡 RELAYS ON")
    else:
        safe_write("OFF\n")
        print("❌ RELAYS OFF")

# ================= TELEGRAM =================
def send_telegram_image(image_path, message):
    global last_telegram_time

    now = time.time()
    if now - last_telegram_time < TELEGRAM_COOLDOWN:
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto"

    try:
        with open(image_path, "rb") as photo:
            r = requests.post(
                url,
                files={"photo": photo},
                data={"chat_id": CHAT_ID, "caption": message}
            )

        if r.status_code == 200:
            last_telegram_time = now
            print("✅ Telegram sent")

    except Exception as e:
        print("Telegram error:", e)

# ================= EMAIL =================
def send_email(image_path):
    msg = EmailMessage()
    msg["Subject"] = "Unknown Person Detected"
    msg["From"] = EMAIL_ADDRESS
    msg["To"] = RECEIVER_EMAIL
    msg.set_content("Unknown detected")

    try:
        with open(image_path, "rb") as f:
            msg.add_attachment(f.read(), maintype="image", subtype="jpeg")

        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
            smtp.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            smtp.send_message(msg)

        print("📧 Email sent")

    except Exception as e:
        print("Email error:", e)

# ================= LOAD MODEL =================
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer1.yml")

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# ================= FLASK =================
app = Flask(__name__)

def generate_frames():
    global frame_global

    while True:
        if frame_global is None:
            continue

        with lock:
            frame = frame_global.copy()

        _, buffer = cv2.imencode(".jpg", frame)

        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" +
               buffer.tobytes() + b"\r\n")

@app.route("/")
def video_feed():
    return Response(generate_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

def run_flask():
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)

# ================= CAMERA THREAD =================
def camera_thread():
    global frame_global, cap

    while True:
        if cap is None:
            find_camera()
            time.sleep(2)
            continue

        ret, frame = cap.read()

        if not ret:
            print("⚠️ Camera lost, reconnecting...")
            cap.release()
            cap = None
            time.sleep(2)
            continue

        frame = cv2.resize(frame, (640, 480))

        with lock:
            frame_global = frame

# ================= MAIN LOOP =================
def main():
    global last_face_time, unknown_triggered

    print("🚀 System Starting...")

    connect_arduino()
    find_camera()

    while True:

        if frame_global is None:
            continue

        with lock:
            frame = frame_global.copy()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        now = time.time()
        face_detected = len(faces) > 0

        # RELAY LOGIC
        if face_detected:
            last_face_time = now
            set_relays(True)
            unknown_triggered = False
        else:
            if now - last_face_time > FACE_TIMEOUT:
                set_relays(False)

        # FACE CHECK
        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]

            try:
                label, confidence = recognizer.predict(face)
            except:
                continue

            if confidence > CONFIDENCE_THRESHOLD and not unknown_triggered:

                if not os.path.exists("captures"):
                    os.makedirs("captures")

                now_ph = datetime.now(PH_TIMEZONE)
                timestamp = now_ph.strftime("%Y-%m-%d %H:%M:%S")
                file_time = now_ph.strftime("%Y%m%d_%H%M%S")

                img_path = f"captures/unknown_{file_time}.jpg"
                cv2.imwrite(img_path, frame)

                send_telegram_image(img_path, f"Unknown detected\nTime: {timestamp}")
                send_email(img_path)

                unknown_triggered = True

        time.sleep(0.03)

# ================= RUN =================
if __name__ == "__main__":
    threading.Thread(target=camera_thread, daemon=True).start()
    threading.Thread(target=run_flask, daemon=True).start()
    main()
