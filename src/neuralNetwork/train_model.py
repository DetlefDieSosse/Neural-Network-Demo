import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, Input # type: ignore
from tensorflow.keras.callbacks import EarlyStopping # type: ignore

# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalize the data
x_train, x_test = x_train / 255.0, x_test / 255.0

# Expand dimensions to match the input shape of the model
x_train = np.expand_dims(x_train, axis=-1)
x_test = np.expand_dims(x_test, axis=-1)

def build_and_train_model():
    # Define the model architecture
    model = models.Sequential([
        Input(shape=(28, 28, 1)),

        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        
        layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0001)),
        layers.Dropout(0.3),
        layers.Dense(10, activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Early stopping to prevent overfitting
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    
     # Train the model
    model.fit(x_train, y_train, batch_size=32, epochs=50, validation_data=(x_test, y_test), callbacks=[early_stopping])

    # Evaluate the model
    test_acc = model.evaluate(x_test, y_test, verbose=2)
    print('\nTest accuracy:', test_acc)

    model.save('mnist_model.keras')

if __name__ == "__main__":
    print(f'Training data shape: {x_train.shape}, Training labels shape: {y_train.shape}')
    print(f'Testing data shape: {x_test.shape}, Testing labels shape: {y_test.shape}')
    
    build_and_train_model()