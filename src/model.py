import tensorflow as tf
from tensorflow.keras import models, layers

IMAGE_SIZE = 256
CHANNELS = 3

def build_model(n_classes):
    model = models.Sequential([
        tf.keras.layers.Rescaling(1./255),

        layers.Conv2D(32, (3,3), activation='relu'),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D((2, 2)),

        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(n_classes, activation='softmax'),
    ])

    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy']
    )

    return model