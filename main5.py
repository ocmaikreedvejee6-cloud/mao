import cv2
import numpy as np
import serial
import time
import requests
import smtplib
import os
from datetime import datetime, timedelta, timezone
from email.message import EmailMessage
from flask import Flask, Response
import threading

# ================= CONFIG =================
SERIAL_PORT = '/dev/ttyUSB0'
BAUD_RATE = 9600

# ðŸ” PUT YOUR OWN VALUES HERE
TELEGRAM_TOKEN = "8490765768:AAFU-Vpi0HAiS5_2V2mcboWYeiG8W4neiVE"
CHAT_ID = "7175315173"

CONFIDENCE_THRESHOLD = 50
FACE_TIMEOUT = 3
TELEGRAM_COOLDOWN = 30

# ðŸ” EMAIL (USE APP PASSWORD ONLY)
EMAIL_ADDRESS = "growpfiveim312@gmail.com"
EMAIL_PASSWORD = "qerlwnbhfcaprcll"
RECEIVER_EMAIL = "ocmaikreedvejee6@gmail.com"

RASPBERRY_PI_IP = "192.168.1.246"
STREAM_URL = f"http://{RASPBERRY_PI_IP}:5000"

PH_TIMEZONE = timezone(timedelta(hours=8))

# ================= STATES =================
last_face_time = 0
last_telegram_time = 0
unknown_triggered = False
relay_state = False

frame_global = None
lock = threading.Lock()

# ================= LOAD MODEL =================
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer1.yml")

label_map = np.load("labels1.npy", allow_pickle=True).item()

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# ================= ARDUINO =================
arduino = serial.Serial(SERIAL_PORT, BAUD_RATE)
time.sleep(2)

# ================= CAMERA =================
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

# ================= RELAY =================
def set_relays(state):
    global relay_state

    if relay_state == state:
        return

    relay_state = state

    if state:
        arduino.write(b'ON\n')
        print("ðŸ’¡ RELAYS ON")
    else:
        arduino.write(b'OFF\n')
        print("âŒ RELAYS OFF")

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
                data={
                    "chat_id": CHAT_ID,
                    "caption": message + f"\nðŸ“¡ Live: {STREAM_URL}"
                }
            )

        if r.status_code == 200:
            last_telegram_time = now
            print("âœ… Telegram sent")

    except Exception as e:
        print("Telegram error:", e)

# ================= EMAIL =================
def send_email(image_path):
    msg = EmailMessage()
    msg["Subject"] = "Unknown Person Detected"
    msg["From"] = EMAIL_ADDRESS
    msg["To"] = RECEIVER_EMAIL

    msg.set_content(f"Unknown detected.\nLive: {STREAM_URL}")

    try:
        with open(image_path, "rb") as f:
            msg.add_attachment(f.read(), maintype="image", subtype="jpeg")

        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
            smtp.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            smtp.send_message(msg)

        print("ðŸ“§ Email sent")

    except Exception as e:
        print("Email error:", e)

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
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")

@app.route("/")
def video_feed():
    return Response(generate_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

def run_flask():
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)

# ================= CAMERA THREAD =================
def camera_thread():
    global frame_global

    while True:
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, (640, 480))
            with lock:
                frame_global = frame

# ================= MAIN LOOP =================
def main():
    global last_face_time, unknown_triggered

    print("ðŸš€ System Running...")

    while True:
        if frame_global is None:
            continue

        with lock:
            frame = frame_global.copy()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        now = time.time()
        face_detected = len(faces) > 0

        # ================= RELAY =================
        if face_detected:
            last_face_time = now
            set_relays(True)
            unknown_triggered = False
        else:
            if now - last_face_time > FACE_TIMEOUT:
                set_relays(False)

        # ================= FACE CHECK =================
        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]

            try:
                label, confidence = recognizer.predict(face)
            except:
                continue

            if confidence > CONFIDENCE_THRESHOLD:
                if not unknown_triggered:

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
