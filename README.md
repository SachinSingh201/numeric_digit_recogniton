# Numeric Digit Recognition System using CNN

## Introduction
This project implements a numeric digit recognition system using a Convolutional Neural Network (CNN) model. The system is designed to recognize handwritten digits (0-9) from the MNIST dataset.

## Prerequisites
Before running the code, ensure you have the following software and libraries installed:
- Python 3.6+ 
- TensorFlow (python 3.13 does not support tensor flow)
- Keras
- NumPy
- Matplotlib

You can install the required libraries using pip:


## Dataset
The MNIST dataset, a benchmark in the field of machine learning, contains 60,000 training images and 10,000 testing images of handwritten digits.

## Model Architecture
The CNN model consists of the following layers:
- Input Layer: 28x28 grayscale image
- Convolutional Layer: 32 filters, kernel size 3x3, ReLU activation
- Max Pooling Layer: pool size 2x2
- Convolutional Layer: 64 filters, kernel size 3x3, ReLU activation
- Max Pooling Layer: pool size 2x2
- Flatten Layer
- Dense Layer: 128 neurons, ReLU activation
- Output Layer: 10 neurons (one for each digit), softmax activation

## Training the Model
To train the model, run the following command:

The training script  loads the MNIST dataset, builds the CNN model, trains it on the dataset, and saves the trained model to a file.

## Testing the Model
To test the model's performance on the MNIST dataset.




## Usage
To use the digit recognition system, you can provide an image of a handwritten digit to the model for prediction. The `predict.py` script demonstrates how to load an image, preprocess it, and use the trained model to predict the digit:


## Results
The trained CNN model achieves an accuracy of over 99% on the MNIST test dataset.

## Conclusion
This project demonstrates the use of Convolutional Neural Networks for handwritten digit recognition. Feel free to explore and improve the model further!

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments
- The MNIST dataset is provided by Yann LeCun and colleagues.
- The project structure and implementation are inspired by various online tutorials and resources.

## Contact
For any questions or inquiries, please contact [Your Name] at [Your Email].


