🛡️ AI/ML Adversarial Attack Demo — FGSM on MNIST

This project explores security vulnerabilities in machine learning models by demonstrating an adversarial attack using the Fast Gradient Sign Method (FGSM) on a Convolutional Neural Network (CNN) trained on the MNIST dataset.

📌 Project Overview

Machine learning models, especially those used in image classification, are often assumed to be robust. However, adversarial attacks reveal just how easily these models can be fooled with subtle input changes. This project:

Trains a CNN to classify handwritten digits (MNIST)

Performs an FGSM adversarial attack on a test image

Visualises the impact by showing original vs perturbed predictions

The project aims to raise awareness of the security risks in AI/ML systems and promote a security mindset among developers.

🧠 What is FGSM?

Fast Gradient Sign Method (FGSM) is an attack that:

Calculates the gradient of the loss with respect to the input

Uses the sign of the gradient to slightly tweak the input

Causes the model to misclassify while keeping the image visually unchanged

📁 Files Included

fgsm_mnist_demo.py — Main Python script for training and attacking the model

README.md — This file

examples/ — (Optional) Contains screenshots of original and perturbed predictions

📦 Requirements

This project runs on Python 3.8+ and uses:

tensorflow

numpy

matplotlib

You can install the requirements using:

pip install -r requirements.txt

Or manually:

pip install tensorflow numpy matplotlib

🚀 How to Run

Clone the repo:

git clone https://github.com/your-username/fgsm-mnist-demo.git
cd fgsm-mnist-demo


Run the script:

python fgsm_mnist_demo.py


The model will be trained, and then the attack will be applied on a test image.
A side-by-side plot will show:

The original image + prediction

The perturbed image + fooled prediction

📊 Example Output
Original Image	Perturbed Image
Predicted: 7	Predicted: 2

Even though the image looks nearly identical to humans, the model misclassifies it.

🔐 Security Implications

Adversarial attacks like FGSM are not just academic — they can have real-world consequences in:

Medical imaging

Facial recognition

Autonomous vehicles

Agricultural systems

This project is a reminder that AI systems need to be secure, not just accurate.

🧪 Future Work

Explore other attack methods (PGD, DeepFool)

Apply to more complex datasets (CIFAR-10, ImageNet)

Implement basic defense techniques (e.g., adversarial training)

🧾 References

Goodfellow, I., Shlens, J., & Szegedy, C. (2014). Explaining and Harnessing Adversarial Examples

TensorFlow Tutorials

Viso AI on Adversarial ML

🤝 License

MIT License – Feel free to use, modify, and share this project. Attribution appreciated!

🙋‍♂️ Author

Created by [Rayaan Choudhry]
Security Engineering @ UNSW | Interested in AI, Adversarial ML, and Cybersecurity

Feel free to reach out or fork the project to explore further!

Or manually:

pip install tensorflow numpy matplotlib
