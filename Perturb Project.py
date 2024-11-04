import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np

# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalise the images to a 0-1 range
x_train, x_test = x_train / 255.0, x_test / 255.0

# Reshape the data to add a channel dimension
# Numebrs are batch sise (automatic), height and width
# then channel dimension (one colour channel as is greyscale)
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# Define the model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))

# Evaluate the model on the test dataset
test_loss, test_acc = model.evaluate(x_test, y_test)
print('Test accuracy:', test_acc)

# Choose an image from the test set
index = 0
image = x_test[index].reshape(1, 28, 28, 1)         # Reshape the image for the model
true_label = y_test[index]

# Convert image and true_label to tensors
image = tf.convert_to_tensor(image)
true_label = tf.convert_to_tensor([true_label])     # Wrap true_label in a list to make it a tensor of shape (1,)

# Predict on the original image
original_prediction = model.predict(image)
original_predicted_label = np.argmax(original_prediction)

# Display the original image and prediction
plt.subplot(1, 2, 1)
plt.title(f'Original Image\nTrue: {true_label.numpy()[0]}, Pred: {original_predicted_label}')
plt.imshow(image[0].numpy().reshape(28, 28), cmap='gray')

# Define the loss function for the attack
loss_object = tf.keras.losses.SparseCategoricalCrossentropy()

# Compute the gradient of the loss w.r.t the input image
with tf.GradientTape() as tape:
    tape.watch(image)
    prediction = model(image)
    loss = loss_object(true_label, prediction)

# Get the gradient and apply FGSM perturbation
gradient = tape.gradient(loss, image)
epsilon = 0.2  # Small value for perturbation
perturbed_image = image + epsilon * tf.sign(gradient)

# Clip the perturbed image to keep pixel values between 0 and 1
perturbed_image = tf.clip_by_value(perturbed_image, 0, 1)

# Predict on the perturbed image
perturbed_prediction = model.predict(perturbed_image)
perturbed_predicted_label = np.argmax(perturbed_prediction)

# Display the perturbed image and prediction
plt.subplot(1, 2, 2)
plt.title(f'Perturbed Image\nTrue: {true_label.numpy()[0]}, Pred: {perturbed_predicted_label}')
plt.imshow(perturbed_image[0].numpy().reshape(28, 28), cmap='gray')

# Show both original and perturbed images side by side
plt.show()

# Print predictions
print("Original prediction:", original_predicted_label)
print("Prediction after attack:", perturbed_predicted_label)