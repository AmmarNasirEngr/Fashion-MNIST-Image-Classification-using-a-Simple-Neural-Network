##
### Fashion-MNIST-Image-Classification-using-a-Simple-Neural-Network
This notebook demonstrates a basic end-to-end workflow for image classification using a simple Artificial Neural Network (ANN) on the Fashion MNIST dataset. It covers data loading, preprocessing, model definition, training, and evaluation in PyTorch. It has evolved through several key stages, starting with basic data preparation and culminating in training a Artificial Neural Network for Fashion MNIST image classification. Here's a breakdown of the overall flow:

## Overall Code Flow and Training Process

# Initial Setup and Data Loading:
The notebook began by importing essential PyTorch, pandas, and scikit-learn libraries. A random seed was set for reproducibility.
The fmnist_small.csv dataset was loaded, and a visual inspection of the first few images was performed to understand the data's raw format.

# Data Preprocessing for Neural Networks:
The raw pixel data (X) and labels (y) were separated. The dataset was then split into training and testing sets (X_train, X_test, y_train, y_test).
Crucially, pixel values (originally 0-255) were normalized to the 0-1 range by dividing by 255.0. This step is vital for improving neural network training stability and performance.

# Custom PyTorch Dataset and DataLoaders:
A CustomDataset class was implemented to efficiently handle the features and labels as PyTorch tensors. This class facilitates easy access to individual data samples.
DataLoader objects were created for both the training and testing datasets. These loaders are responsible for batching the data and shuffling the training data, which helps in generalizing the model and preventing overfitting.

# Neural Network Model Definition (Evolution from ANN to CNN):
Initially, a simple MyNN (Artificial Neural Network) was defined with fully connected layers. This model served as a baseline, processing flattened image data.
Recognizing the strengths of CNNs for image tasks, the architecture was later updated to CNNModel. This new model incorporates:
Reshaping: The input (batch_size, 784) vector is reshaped to (batch_size, 1, 28, 28) to represent grayscale images.
Convolutional Layers (nn.Conv2d): These layers automatically learn spatial hierarchies and features from the images.
ReLU Activation Functions: Applied after convolutional and linear layers for non-linearity.
Max Pooling Layers (nn.MaxPool2d): Used to reduce the spatial dimensions, providing translational invariance and reducing computational load.
Fully Connected Layers (nn.Linear): After convolutional and pooling layers, the data is flattened and passed through dense layers for final classification.

# Training Configuration:
Key hyperparameters were set: epochs (100 iterations over the training data) and learning_rate (0.1 for controlling step size during optimization).
nn.CrossEntropyLoss was chosen as the loss function, appropriate for multi-class classification problems.
optim.SGD (Stochastic Gradient Descent) was used as the optimizer, responsible for updating the model's weights based on the calculated gradients of the loss.

# Model Training Loop:
The model underwent a training process spanning 100 epochs. In each epoch, the model iterates through the train_loader, processing data in batches.
For every batch: a forward pass computes predictions, the loss is calculated, gradients are computed via backpropagation, and the optimizer updates the model's parameters.
The average loss per epoch was printed, showing a general decrease over time, indicating that the model was learning from the training data.

# Model Evaluation:
After training, the model was explicitly set to evaluation mode (model.eval()). This step is important as it disables certain layers (like Dropout, if used) that behave differently during training versus inference.
The model's performance was then evaluated on the test_loader (unseen data) without gradient computations (torch.no_grad()).
The final accuracy was calculated by comparing the model's predictions to the true labels. This metric provides an indication of how well the model generalizes to new, unobserved data.

# By systematically moving through these steps, from raw data to a trained and evaluated CNN, the notebook demonstrates a complete machine learning pipeline for image classification, with an iterative improvement from a simpler ANN to a more powerful CNN architecture.
