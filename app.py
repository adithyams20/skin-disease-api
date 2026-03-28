import os
import json
import sqlite3
import numpy as np
import requests
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# ==============================
# DOWNLOAD MODEL IF NOT EXISTS
# ==============================
MODEL_PATH = "skin_disease_model.h5"

if not os.path.exists(MODEL_PATH):
    print("⬇️ Downloading model...")

    FILE_ID = "1yuxX9Fll0ZL6AZ7D1SF0wKFU0atiuAAo"  # 🔥 PUT YOUR FILE ID HERE
    URL = f"https://drive.google.com/uc?export=download&id={FILE_ID}"

    response = requests.get(URL)

    with open(MODEL_PATH, "wb") as f:
        f.write(response.content)

    print("✅ Model downloaded")

# ==============================
# LOAD MODEL
# ==============================
print("Loading model...")
model = load_model(MODEL_PATH)
print("✅ Model loaded")

# ==============================
# LOAD CLASS INDEX
# ==============================
with open("class_indices.json", "r") as f:
    class_indices = json.load(f)

index_to_class = {v: k for k, v in class_indices.items()}

# ==============================
# DB
# ==============================
def get_db():
    conn = sqlite3.connect("history.db")
    conn.row_factory = sqlite3.Row
    return conn

# ==============================
# IMAGE PREPROCESS
# ==============================
def preprocess(path):
    img = load_img(path, target_size=(224, 224))
    img = img_to_array(img) / 255.0
    return np.expand_dims(img, axis=0)

# ==============================
# LOGIN
# ==============================
@app.route('/login', methods=['POST'])
def login():
    try:
        data = request.get_json(force=True)

        conn = get_db()
        user = conn.execute(
            "SELECT * FROM users WHERE username=? AND password=?",
            (data["username"], data["password"])
        ).fetchone()
        conn.close()

        if user:
            return jsonify({
                "user_id": user["id"],
                "username": user["username"]
            })

        return jsonify({"error": "Invalid login"}), 401

    except Exception as e:
        print("LOGIN ERROR:", e)
        return jsonify({"error": "Server error"}), 500

# ==============================
# REGISTER
# ==============================
@app.route('/register', methods=['POST'])
def register():
    try:
        data = request.get_json(force=True)

        conn = get_db()
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM users WHERE username=?", (data["username"],))
        if cursor.fetchone():
            return jsonify({"error": "User exists"})

        cursor.execute(
            "INSERT INTO users (username, password) VALUES (?,?)",
            (data["username"], data["password"])
        )

        conn.commit()
        conn.close()

        return jsonify({"message": "Registered"})

    except Exception as e:
        print("REGISTER ERROR:", e)
        return jsonify({"error": "Registration failed"}), 500

# ==============================
# PREDICT
# ==============================
@app.route('/predict', methods=['POST'])
def predict():
    try:
        print("📥 Request received")

        if 'image' not in request.files:
            return jsonify({"error": "No image"}), 400

        file = request.files['image']

        os.makedirs("uploads", exist_ok=True)
        path = os.path.join("uploads", file.filename)
        file.save(path)

        print("🧠 Processing image...")

        img = preprocess(path)

        preds = model.predict(img)
        idx = int(np.argmax(preds))
        confidence = float(np.max(preds))

        label = index_to_class[idx]

        print("✅ Predicted:", label)

        conn = get_db()
        cursor = conn.cursor()

        disease = cursor.execute(
            "SELECT * FROM disease_info WHERE disease_name=?",
            (label,)
        ).fetchone()

        if not disease:
            return jsonify({"error": "No database entry"}), 500

        # ================= SAFE HISTORY =================
        user_id = request.form.get("user_id")

        if user_id:
            try:
                cursor.execute("PRAGMA table_info(history)")
                columns = [col[1] for col in cursor.fetchall()]

                if "disease_name" in columns:
                    cursor.execute(
                        "INSERT INTO history (user_id, disease_name, confidence, image_path) VALUES (?,?,?,?)",
                        (user_id, label, confidence, path)
                    )

                elif "result" in columns:
                    cursor.execute(
                        "INSERT INTO history (user_id, result, confidence) VALUES (?,?,?)",
                        (user_id, label, confidence)
                    )

                conn.commit()

            except Exception as e:
                print("History error:", e)

        conn.close()

        print("📤 Sending response")

        return jsonify({
            "disease_name": disease["display_name"],
            "confidence": round(confidence * 100, 2),
            "description": disease["description"],
            "recommendation": disease["medical_recommendation"],
            "advice": disease["skincare_advice"]
        })

    except Exception as e:
        print("🔥 ERROR:", e)
        return jsonify({"error": "Prediction failed"}), 500

# ==============================
# HISTORY
# ==============================
@app.route('/history/<int:user_id>', methods=['GET'])
def history(user_id):
    try:
        conn = get_db()
        cursor = conn.cursor()

        cursor.execute("PRAGMA table_info(history)")
        columns = [col[1] for col in cursor.fetchall()]

        if "disease_name" in columns:
            rows = cursor.execute(
                "SELECT disease_name, confidence FROM history WHERE user_id=? ORDER BY id DESC",
                (user_id,)
            ).fetchall()
        elif "result" in columns:
            rows = cursor.execute(
                "SELECT result as disease_name, confidence FROM history WHERE user_id=? ORDER BY id DESC",
                (user_id,)
            ).fetchall()
        else:
            rows = []

        conn.close()

        data = []
        for r in rows:
            data.append({
                "disease_name": r["disease_name"],
                "confidence": round(r["confidence"] * 100, 2)
            })

        return jsonify(data)

    except Exception as e:
        print("HISTORY ERROR:", e)
        return jsonify([])

# ==============================
# RUN
# ==============================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)