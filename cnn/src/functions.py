import keras
from keras.optimizers import SGD

def build_model(learning_rate, n_filters):
    model = keras.models.Sequential([
      keras.layers.Conv2D(n_filters, 7, activation="relu", padding="same", input_shape=(28, 28, 1)),
      keras.layers.MaxPooling2D(2),
      keras.layers.Conv2D(n_filters*2, 3, activation="relu", padding="same"),
      keras.layers.Conv2D(n_filters*2, 3, activation="relu", padding="same"),
      keras.layers.MaxPooling2D(2),
      keras.layers.Flatten(),
      keras.layers.Dense(n_filters*2, activation="relu"),
      keras.layers.Dropout(0.5),
      keras.layers.Dense(n_filters*2, activation="relu"),
      keras.layers.Dropout(0.5),
      keras.layers.Dense(10, activation="softmax")
    ])

    custom_optimizer = SGD(lr = learning_rate)

    model.compile(loss="sparse_categorical_crossentropy", optimizer=custom_optimizer, metrics=["accuracy"])

    return model
