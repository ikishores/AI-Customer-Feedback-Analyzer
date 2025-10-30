# src/sentiment_model.py
import pandas as pd
import numpy as np
import re
import nltk
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Bidirectional, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import os


nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ========== Step 1: Load Data ==========
data_path = "../data/simulated_feedback.csv"
df = pd.read_csv(data_path)

print(" Data Loaded Successfully!")
print(df.head())

# ========== Step 2: Text Cleaning ==========
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = re.sub('[^a-zA-Z]', ' ', text)  
    text = text.lower()
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return ' '.join(words)

df['cleaned_text'] = df['text'].apply(clean_text)


label_encoder = LabelEncoder()
df['label_encoded'] = label_encoder.fit_transform(df['sentiment_label'])
print("\nLabel Encoding Map:", dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))))


X_train, X_test, y_train, y_test = train_test_split(
    df['cleaned_text'], df['label_encoded'], test_size=0.2, random_state=42
)


tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train)

X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

max_len = 30
X_train_pad = pad_sequences(X_train_seq, maxlen=max_len, padding='post', truncating='post')
X_test_pad = pad_sequences(X_test_seq, maxlen=max_len, padding='post', truncating='post')


model = Sequential([
    Embedding(input_dim=5000, output_dim=64, input_length=max_len),
    Bidirectional(LSTM(64, return_sequences=False)),
    Dropout(0.5),
    Dense(32, activation='relu'),
    Dense(3, activation='softmax')
])

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()


early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

history = model.fit(
    X_train_pad, y_train,
    validation_data=(X_test_pad, y_test),
    epochs=10,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1
)


loss, accuracy = model.evaluate(X_test_pad, y_test, verbose=0)
print(f"\n Model Accuracy: {accuracy*100:.2f}%")


plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


os.makedirs("../models", exist_ok=True)
model.save("../models/sentiment_bilstm_model.h5")
print(" Model saved as sentiment_bilstm_model.h5")
