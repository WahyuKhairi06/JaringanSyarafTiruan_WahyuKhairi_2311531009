# =====================================================
# 🌦️ GUI PENGENALAN CUACA MENGGUNAKAN MODEL CNN
# =====================================================

import tkinter as tk
from tkinter import filedialog
from tkinter import Label, Button
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf

# =====================================================
# 1️⃣ Muat Model CNN yang Sudah Dilatih
# =====================================================
model = tf.keras.models.load_model("weather_cnn_model.h5")

# Daftar label cuaca (urut sesuai dataset)
labels = ["cloudy", "foggy", "rainy", "snowy", "sunny"]

# =====================================================
# 2️⃣ Fungsi Prediksi Cuaca
# =====================================================
def predict_weather(img_path):
    try:
        # Preprocessing gambar sesuai input model CNN
        img = Image.open(img_path).resize((128, 128))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)  # (1, 128, 128, 3)
        
        pred = model.predict(img_array)
        idx = np.argmax(pred)
        confidence = pred[0][idx] * 100
        
        return labels[idx].capitalize(), confidence
    
    except Exception as e:
        return f"Error: {e}", 0.0

# =====================================================
# 3️⃣ Fungsi Upload dan Tampilkan Hasil
# =====================================================
def upload_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.jpeg *.png")])
    if not file_path:
        return

    # Tampilkan gambar di GUI
    img = Image.open(file_path).resize((250, 250))
    img_tk = ImageTk.PhotoImage(img)
    label_img.config(image=img_tk)
    label_img.image = img_tk

    # Prediksi cuaca
    hasil, persen = predict_weather(file_path)
    label_result.config(
        text=f"Hasil Prediksi: {hasil}\nKeyakinan: {persen:.2f}%",
        fg="white"
    )

# =====================================================
# 4️⃣ Setup GUI
# =====================================================
root = tk.Tk()
root.title("🌦️ Prediksi Cuaca dengan CNN")
root.geometry("420x520")
root.configure(bg="#1e293b")

Label(root, text="🌤️ Weather Classification (CNN)", 
      font=("Arial", 16, "bold"), fg="white", bg="#1e293b").pack(pady=15)

label_img = Label(root, bg="#1e293b")
label_img.pack(pady=10)

Button(root, text="📁 Upload Gambar Cuaca", command=upload_image,
       bg="#3b82f6", fg="white", font=("Arial", 12, "bold"), 
       relief="ridge", padx=10, pady=5).pack(pady=10)

label_result = Label(root, text="", bg="#1e293b", fg="white", font=("Arial", 13))
label_result.pack(pady=20)

Label(root, text="Made by Wahyu Khairi 💡", 
      bg="#1e293b", fg="#94a3b8", font=("Arial", 10)).pack(side="bottom", pady=10)

root.mainloop()
