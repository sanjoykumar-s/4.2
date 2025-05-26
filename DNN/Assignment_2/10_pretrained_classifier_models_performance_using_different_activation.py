"""*Import Libraries*"""

from tensorflow.keras.datasets.cifar100 import load_data
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Input, Flatten, Dense, Lambda, GlobalAveragePooling2D, Normalization, Dropout
from tensorflow.keras.models import Model
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time

from tensorflow.keras.applications import (VGG16, VGG19, MobileNet, DenseNet121,
                                           DenseNet169, DenseNet201, ResNet101,
                                           ResNet152, ResNet50, VGG19, MobileNetV2)

"""*Display Images*"""

def display_img(img_set, title_set):
  n = len(title_set)
  plt.figure(figsize=(10, 6))
  for i in range(n):
    plt.subplot(3, 3, i + 1)
    plt.imshow(img_set[i], cmap = 'gray')
    plt.title(title_set[i])
  plt.tight_layout()
  plt.show()
  plt.close()

"""*Filter Dataset for 20 classes*"""

def filter_classes(X, Y, num_classes):
    filtered_X = []
    filtered_Y = []
    for i in range(len(Y)):
        if Y[i] < num_classes:
            filtered_X.append(X[i])
            filtered_Y.append(Y[i])
    return np.array(filtered_X), np.array(filtered_Y)

"""*Load Data and Preprocess data*"""

# Load Dataset
(raw_trainX, raw_trainY), (raw_testX, raw_testY) = load_data(label_mode='fine')

# Filter Dataset for only class 20
trainX, trainY = filter_classes(raw_trainX, raw_trainY, 20)
testX, testY = filter_classes(raw_testX, raw_testY, 20)


display_img(trainX[:9], trainY[:9])

# One hot encode the labels
trainY = to_categorical(trainY, 20)
testY = to_categorical(testY, 20)

# Create TensorFlow Dataset
train_dataset = tf.data.Dataset.from_tensor_slices((trainX, trainY))
test_dataset = tf.data.Dataset.from_tensor_slices((testX, testY))

"""List of 10 pre-trained Model"""

models = [
    ('VGG16', VGG16),
    ('VGG19', VGG19),
    ('MobileNet', MobileNet),
    ('MobileNetV2', MobileNetV2),
    ('DenseNet121', DenseNet121),
    ('DenseNet169', DenseNet169),
    ('DenseNet201', DenseNet201),
    ('ResNet50', ResNet50),
    ('ResNet101', ResNet101),
    ('ResNet152', ResNet152)
]

"""PreProcess Data"""

def preprocess_data(image, label, model_name, training=False):

    # Resize the image to the input size of the model
    image = tf.image.resize(image, (224, 224))

    if training:
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_brightness(image, max_delta=0.1)
        image = tf.image.random_contrast(image, lower=0.9, upper=1.1)

    if model_name == 'VGG16':
         image = tf.keras.applications.vgg16.preprocess_input(image)
    elif model_name == 'VGG19':
        image = tf.keras.applications.vgg19.preprocess_input(image)
    elif model_name == 'MobileNet':
        image = tf.keras.applications.mobilenet.preprocess_input(image)
    elif model_name == 'MobileNetV2':
        image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    elif model_name == 'DenseNet121':
        image = tf.keras.applications.densenet.preprocess_input(image)
    elif model_name in ['DenseNet169', 'DenseNet201']:
        image = tf.keras.applications.densenet.preprocess_input(image)
    elif model_name in ['ResNet50', 'ResNet101', 'ResNet152']:
        image = tf.keras.applications.resnet.preprocess_input(image)
    return image, label

from inspect import Parameter
activation_functions = ["softmax", "sigmoid", "relu", "tanh"]
results = []

for model_name, backbone_model in models:
    for activation_fn in activation_functions:
        print(f"\nTraining {model_name} with {activation_fn} activation...")

        train_ds = train_dataset.map(lambda image, label: preprocess_data(image, label, model_name, training=True))
        train_ds = train_ds.shuffle(buffer_size=1000).batch(batch_size=32).prefetch(buffer_size=1)
        test_ds = test_dataset.map(lambda image, label: preprocess_data(image, label, model_name, training=False))
        test_ds = test_ds.batch(batch_size=32).prefetch(buffer_size=1)

        #backbone model
        backbone = backbone_model(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        backbone.trainable = False

        x = backbone.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(128, activation='relu')(x)
        outputs = Dense(20, activation = activation_fn)(x)
        model = Model(inputs=backbone.input, outputs=outputs)

        # Compile the model
        model.compile(optimizer= 'adam', loss='categorical_crossentropy', metrics=['accuracy'])

        # Train the model
        model.fit(train_ds, epochs=5, validation_data=test_ds, verbose=1)

        # Evaluate the model
        test_loss, test_accuracy = model.evaluate(test_ds, verbose=0)

        # Mesure inference time for one test sample
        sample_test = np.expand_dims(testX[0], axis=0)
        sample_test = tf.image.resize(sample_test, (224, 224))
        start_time = time.time()
        model.predict(sample_test)
        inference_time = time.time() - start_time
        inference_time *= 1000  # Convert to milliseconds

        # Get the number of parameter
        num_Parameter = backbone.count_params()

        # get the num_layers
        num_layers = len(backbone.layers)

        # Store results
        results.append({
            'Model': model_name,
            'Activation Function': activation_fn,
            'Test Accuracy': test_accuracy,
            'Inference Time (ms)': inference_time,
        })

"""Show the Performance Table for each model and for each activation *function*"""

# Convert results to DataFrame
results_df = pd.DataFrame(results, columns=['Model', 'Activation Function', 'Test Accuracy', 'Inference Time (ms)'])
# Display results
print("\nResults:")
print(results_df)