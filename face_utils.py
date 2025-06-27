import cv2
import numpy as np

CASCADE_PATH = "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + CASCADE_PATH)


def extract_face_from_image(image_stream):
    
    nparr = np.frombuffer(image_stream.read(), np.uint8)
    img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img_np is None:
        return None, 

    gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        return None, 

    (x, y, w, h) = faces[0]
    face_img = gray[y:y + h, x:x + w]

    return face_img, None


def encode_image_to_bytes(img):
    
    ret, jpeg = cv2.imencode(".jpg", img)
    if not ret:
        return None
    return jpeg.tobytes()


def decode_bytes_to_image(blob_bytes):
    
    nparr = np.frombuffer(blob_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    return img


def compare_faces(face1, face2, threshold=70):
    
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train([face2], np.array([0]))  

    label, confidence = recognizer.predict(face1)

    return confidence < threshold
