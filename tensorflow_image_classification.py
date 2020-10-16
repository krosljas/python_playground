import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)

# Import and load Fashion MNIST data directly from tensorflow
fashion_mnist = tf.keras.datasets.fashion_mnist

# Loading the dataset returns four NumPy arrays

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
# First ruple of arrays contains the data the model uses to learn
# Second tuple of arrays contains the data that is tested against the model

# Labels are are arrays of integers, 0 to 9, corresponding to the class of clothing the image represents
class_names = ['T-shirt/top', 'Trouser', 'Pullover',
               'Dress', 'Coat', 'Sandal', 'Shirt', 
               'Sneaker', 'Bag', 'Ankle boot']

# Image pixel values fall in the range of 0 to 255
# Scale these values to a range of 0 to 1 before feeding them to the neural network model
train_images = train_images / 255.0
test_images = test_images / 255.0

# Build the neural network

# Set up the layers - layers attempt to extract representations from
# the data fed into them. Hopefully these representations are meaningful

# Chain together layers to implement deep learning
# We will be using a sequential model
# Such a model is appropriate for a plain stack of layers where each layer
# has exactly one input tensor and one output tensor
model = tf.keras.Sequential([
        
        # First layer reformats data from 2-D array to a 1-D array of 28 * 28 = 784 pixels
        # Simply, this layer flattens the pixels by unstacking rows of pixels in the image and lining them up
        tf.keras.layers.Flatten(input_shape=(28,28)),
        
        # Network consists of a sequence of two Dense layers
        # These are densely connected neural layers
        
        # The Dense layer has 128 nodes
        # Each node contains a score that indicates the current image belongs to one of the 10 classes
        tf.keras.layers.Dense(128, activation='relu'),
        # The last layer returns logits array with length of 10
        tf.keras.layers.Dense(10)
        
    ])

# Before the model is ready for training, additional settings must be adder
# These are added during the model's complile step

# Loss Function - Measures how accurate the model is during training. Miminimize this function to "steer" model in the right direction
# Optimizer - This is how the model is updated based on the data it sees and its loss function
# Metrics - Used to monitor the training and testing steps. Accuracy is the fraction of the images that are correctly classified
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Train the neural network by Feeding the training data (images and labels) to the model
# To start training the model, call model.fit
model.fit(train_images, train_labels, epochs=10)
# As the model trains, the loss and accuracy metrics are displayed
# Accuracy reaches approximately 91%

# Compare how the model performs on the test dataset
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print('\nTest accuracy:', test_acc)
# Model is overfitted - less accurate than the training dataset
# OVerfitting happens when a model performs worse on new (previosly unseen) inputs than it does on the training data
# Will attempt to prevent overfitting upon successful creation of this model

# Make predictions
# Add a softmax layer to convert the model's output of logits to probabilities
probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
# Here the model has predicted the label for each image in the testing set
predictions = probability_model.predict(test_images)
# A prediction in this array is an array of 10 numbers, representing the models "confidence" in its prediction
# Take index of highest confidence and correspond it to the class_names array
print(np.argmax(predictions[0]))
# Compare it to the actual label attached to that test image
print(test_labels[0])

def plot_image(i, predictions_array, true_label, img):
  true_label, img = true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  plt.imshow(img, cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'

  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
  true_label = true_label[i]
  plt.grid(False)
  plt.xticks(range(10))
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)

  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')
  

num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i, predictions[i], test_labels, test_images)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, predictions[i], test_labels)
plt.tight_layout()
plt.show()






