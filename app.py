from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import cv2
import numpy as np
import cx_Oracle

dsn = cx_Oracle.makedsn("192.168.0.12", 1522, service_name="PROD18")
connection = cx_Oracle.connect(user="SIASECO_V2", password="SIASECO_V2", dsn=dsn)

CASCADE_PATH = "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + CASCADE_PATH)
recognizer = cv2.face.LBPHFaceRecognizer_create()

app = Flask(__name__)
CORS(app)
os.makedirs("face_data", exist_ok=True)

def extract_face_from_image(image_file):
    nparr = np.frombuffer(image_file.read(), np.uint8)
    img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img_np is None:
        return None, "Invalid image file"

    gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    if len(faces) == 0:
        return None, "No face detected in the image"

    (x, y, w, h) = faces[0]
    face_img = gray[y:y + h, x:x + w]
    return face_img, "Face extracted"

@app.route("/register", methods=["POST"])
def register():
    image = request.files.get("faceimage")
    user_id = request.form.get("userId")
    username = request.form.get("username")
    application_type = request.form.get("application_type")

    if not image or not user_id or not username or not application_type:
        return jsonify({"error": "Missing userId, username, application_type, or face image"}), 400

    image.seek(0)
    face_img, msg = extract_face_from_image(image)
    if face_img is None:
        return jsonify({"error": msg}), 400

    image.seek(0)
    img_data = image.read()

    try:
        cursor = connection.cursor()
        cursor.execute("""
            INSERT INTO users (user_id, username, face_image, application_type)
            VALUES (:1, :2, :3, :4)
        """, (user_id, username, img_data, application_type))
        connection.commit()
        cursor.close()
    except Exception as e:
        return jsonify({"error": f"Database error: {str(e)}"}), 500

    return jsonify({
        "message": f"User '{username}' registered successfully",
        "user": {"userId": user_id, "username": username, "application_type": application_type}
    }), 200

@app.route("/login", methods=["POST"])
def login():
    image = request.files.get("faceimage")
    if not image:
        return jsonify({"error": "Image is required"}), 400

    image.seek(0)
    input_face, msg = extract_face_from_image(image)
    if input_face is None:
        return jsonify({"error": msg}), 400

    try:
        cursor = connection.cursor()
        cursor.execute("SELECT user_id, username, face_image, application_type FROM users")
        rows = cursor.fetchall()
        cursor.close()
    except Exception as e:
        return jsonify({"error": f"Database read error: {str(e)}"}), 500

    faces = []
    labels = []
    label_to_user = {}
    label_id = 0

    for user_id, username, face_blob, application_type in rows:
        blob_bytes = face_blob.read()
        db_nparr = np.frombuffer(blob_bytes, np.uint8)
        db_img = cv2.imdecode(db_nparr, cv2.IMREAD_COLOR)
        if db_img is None:
            continue
        db_gray = cv2.cvtColor(db_img, cv2.COLOR_BGR2GRAY)
        db_faces = face_cascade.detectMultiScale(db_gray, 1.3, 5)
        if len(db_faces) == 0:
            continue
        (x, y, w, h) = db_faces[0]
        db_face = db_gray[y:y+h, x:x+w]

        if db_face.shape != input_face.shape:
            db_face = cv2.resize(db_face, (input_face.shape[1], input_face.shape[0]))

        faces.append(db_face)
        labels.append(label_id)
        label_to_user[label_id] = {
            "userId": user_id,
            "username": username,
            "application_type": application_type
        }
        label_id += 1

    if not faces:
        return jsonify({"error": "No face data available in database"}), 500

    recognizer.train(faces, np.array(labels))
    label, confidence = recognizer.predict(input_face)

    if confidence < 70:
        matched_user = label_to_user.get(label)
        return jsonify({"message": "Login successful", "user": matched_user}), 200
    else:
        return jsonify({"message": "Login failed", "error": "Face not recognized"}), 401

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0",port=5000,ssl_context=("./certificate/certificate.crt", "./certificate/certificate.key"))
