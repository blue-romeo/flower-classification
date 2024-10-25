# Project Overview:

This project involves fine-tuning a pre-trained ResNet-18 model on the CIFAR-10 dataset to classify images of flowers into five classes: roses, dandelion, tulips, sunflowers, and daisy. The fine-tuned model can then be used to predict the class of a given flower image.

# Dependencies:

* Python 3.x
* PyTorch
* torchvision
* NumPy
* Matplotlib (for visualization, optional)
# Dataset:

The dataset used was sourced from kaggle: https://www.kaggle.com/datasets/rahmasleam/flowers-dataset
## About the dataset
### Flowers Dataset

This dataset consists of images from five distinct flower species, ideal for tasks like image classification and computer vision projects. It provides a diverse range of floral images, enabling models to learn the subtle differences between species.

* Daisy: Known for its simple, classic white petals and yellow center.
* Dandelion: Bright yellow flowers that are common in fields and gardens.
* Roses: The quintessential symbol of love and beauty, varying in shades of red, pink, and other colors.
* Sunflowers: Large, sun-like blooms, recognized for their vibrant yellow petals and central brown disc.
* Tulips: Elegant and colorful blooms, popular in gardens and floral arrangements.
This dataset is a great resource for building models capable of recognizing and differentiating between various species of flowers.
# Model:

A pre-trained ResNet-18 model on the CIFAR-10 dataset.
The model's final layer will be replaced with a new layer with the appropriate number of output classes (5 in this case).
# Training:

* Load the pre-trained ResNet-18 model.
* Replace the final layer with a new layer with 5 output units.
* Load the flower image dataset.
* Define data loaders for training and validation sets.
* Set up the optimizer and loss function.
* Train the model using appropriate hyperparameters (e.g., learning rate, batch size, number of epochs).
* Save the trained model.
# Prediction:

* Load the trained model.
* Preprocess the input image (e.g., resize, normalize).
* Pass the preprocessed image through the model.
* Get the predicted class label based on the model's output.
# Evaluation:

Evaluate the model's performance on a separate validation or test set using metrics such as accuracy and loss.
# Usage:

* Install the required dependencies.
* Prepare your dataset.
* Run the training script.
* Use the trained model for prediction on new flower images.
# Future Considerations:

* Consider data augmentation techniques (e.g., rotation, flipping, cropping) to improve model generalization.
* Experiment with different hyperparameters to optimize performance.
* Explore transfer learning techniques to fine-tune the model on a smaller dataset.
* Visualize the model's predictions to understand its strengths and weaknesses.

# REFERENCES

1.   https://rumn.medium.com/custom-pytorch-image-classifier-from-scratch-d7b3c50f9fbe
2.   https://huggingface.co/learn/computer-vision-course/unit2/cnns/resnet

