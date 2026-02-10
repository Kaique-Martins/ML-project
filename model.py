import tensorflow as tf
from tensorflow.keras import layers, models

def build_model(input_shape, n_classes=1):
    model = models.Sequential([
        layers.Input(shape=(input_shape,)),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(64, activation='relu'),
        layers.Dense(n_classes, activation='sigmoid' if n_classes==1 else 'softmax')
    ])
    loss = 'binary_crossentropy' if n_classes==1 else 'sparse_categorical_crossentropy'
    model.compile(optimizer='adam', loss=loss, metrics=['accuracy'])
    return model
