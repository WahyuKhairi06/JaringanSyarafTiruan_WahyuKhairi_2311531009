# =====================================================
# TRAINING MODEL CNN (Convolutional Neural Network)
# UNTUK KLASIFIKASI CITRA CUACA
# =====================================================

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# =====================================================
# 1️⃣ Pengaturan Awal
# =====================================================
data_dir = "dataset"  # folder berisi 5 kelas cuaca
img_size = (128, 128)  # lebih besar agar CNN bisa ekstraksi fitur lebih baik
batch_size = 32
epochs = 25

# =====================================================
# 2️⃣ Data Preprocessing + Augmentasi
# =====================================================
datagen = ImageDataGenerator(
    rescale=1.0/255,
    validation_split=0.2,
    rotation_range=30,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True
)

train_gen = datagen.flow_from_directory(
    data_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

val_gen = datagen.flow_from_directory(
    data_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

# =====================================================
# 3️⃣ Bangun Arsitektur CNN
# =====================================================
model = Sequential([
    # --- Blok 1
    Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)),
    MaxPooling2D(2,2),

    # --- Blok 2
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    # --- Blok 3
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(train_gen.num_classes, activation='softmax')
])

# Kompilasi model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# =====================================================
# 4️⃣ Training Model CNN
# =====================================================
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=epochs,
    verbose=1
)

# =====================================================
# 5️⃣ Evaluasi Model
# =====================================================
loss, acc = model.evaluate(val_gen)
print(f"\n🎯 Akurasi Validasi: {acc*100:.2f}%")

# --- Confusion Matrix ---
y_true = val_gen.classes
y_pred = np.argmax(model.predict(val_gen), axis=1)

cm = confusion_matrix(y_true, y_pred)
labels = list(val_gen.class_indices.keys())

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
plt.xlabel("Prediksi")
plt.ylabel("Asli")
plt.title("Confusion Matrix - CNN Cuaca")
plt.show()

# --- Classification Report ---
print("\n📊 Classification Report:")
print(classification_report(y_true, y_pred, target_names=labels))

# =====================================================
# 6️⃣ Simpan Model
# =====================================================
model.save("weather_cnn_model.h5")
print("✅ Model CNN disimpan sebagai 'weather_cnn_model.h5'")
