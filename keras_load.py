import tensorflow as tf


# Definiáljuk a fájl elérési útját
model_path = r'saved_model\cnn_rnn_model.keras'

# Betöltjük a modellt
model = tf.keras.models.load_model(model_path)

# Ellenőrizzük a modell architektúráját
model.summary()