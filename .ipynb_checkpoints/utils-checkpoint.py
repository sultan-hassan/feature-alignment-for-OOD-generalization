import torch 
import torch.nn as nn
import torch.nn.functional as F
import ot # package to compute optimal transport loss (https://pythonot.github.io/)

class SimpleCNN(nn.Module):
    """
    A simple Convolutional Neural Network (CNN) for image classification.

    This network consists of two convolutional layers followed by two fully connected layers.
    Each convolutional layer is followed by ReLU activation and max-pooling. The fully connected
    layers include ReLU activations and produce the final classification logits with a log-softmax output.

    Attributes:
        conv1 (nn.Conv2d): First convolutional layer with 1 input channel, 4 output channels,
                          and a 3x3 kernel.
        conv2 (nn.Conv2d): Second convolutional layer with 4 input channels, 8 output channels,
                          and a 3x3 kernel.
        fc1 (nn.Linear): Fully connected layer with 200 input features and 512 output features.
        fc2 (nn.Linear): Fully connected layer with 512 input features and 10 output features
                         (corresponding to 10 classes).
    """

    def __init__(self):
        """
        Initializes the SimpleCNN model by defining the layers.
        """
        super().__init__()
        self.conv1 = nn.Conv2d(1, 4, kernel_size=3)  # First convolutional layer
        self.conv2 = nn.Conv2d(4, 8, kernel_size=3)  # Second convolutional layer
        self.fc1 = nn.Linear(200, 512)              # First fully connected layer
        self.fc2 = nn.Linear(512, 10)               # Second fully connected layer

    def forward(self, x):
        """
        Defines the forward pass of the SimpleCNN model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 1, height, width).

        Returns:
            list: A list containing the intermediate feature maps:
                  [h_conv1, h_conv2, h_fc1, h_fc2].
            torch.Tensor: The final output tensor with log-softmax applied, of shape (batch_size, 10).
        """
        # First convolutional layer: Convolution -> ReLU -> Max Pooling
        h_conv1 = self.conv1(x)                      # Apply first convolutional layer
        h_conv1 = F.relu(h_conv1)                    # Apply ReLU activation
        h_conv1 = F.max_pool2d(h_conv1, 2)           # Apply max pooling with kernel size 2

        # Second convolutional layer: Convolution -> ReLU -> Max Pooling
        h_conv2 = self.conv2(h_conv1)                # Apply second convolutional layer
        h_conv2 = F.relu(h_conv2)                    # Apply ReLU activation
        h_conv2 = F.max_pool2d(h_conv2, 2)           # Apply max pooling with kernel size 2

        # Flatten the feature maps for the fully connected layers
        h_fc1 = torch.flatten(h_conv2, 1)            # Flatten feature maps into a 2D tensor

        # First fully connected layer: Linear -> ReLU
        h_fc1 = self.fc1(h_fc1)                      # Apply first fully connected layer
        h_fc1 = F.relu(h_fc1)                        # Apply ReLU activation

        # Second fully connected layer: Linear -> Log-Softmax
        h_fc2 = self.fc2(h_fc1)                      # Apply second fully connected layer
        output = F.log_softmax(h_fc2, dim=1)         # Apply log-softmax for classification output

        # Return intermediate features and final output
        return [h_conv1, h_conv2, h_fc1, h_fc2], output


class add_gaussian_noise():
    """
    A callable class for adding Gaussian noise to input images.

    This class adds random noise sampled from a Gaussian distribution with the specified mean
    and standard deviation to the input tensor. This is 

    Attributes:
        mean (float): The mean of the Gaussian noise to be added. Default is 0.
        std (float): The standard deviation of the Gaussian noise to be added. Default is 1.0
    """

    def __init__(self, mean=0., std=1.0):
        """
        Initializes the add_gaussian_noise class with the specified mean and standard deviation.

        Args:
            mean (float, optional): Mean of the Gaussian noise. Default is 0.
            std (float, optional): Standard deviation of the Gaussian noise. Default is 1.0.
        """
        self.std = std
        self.mean = mean

    def __call__(self, image):
        """
        Adds Gaussian noise to the input image.

        Args:
            image (torch.Tensor): Input tensor representing the image. The shape is typically
                                  (channels, height, width) or (batch_size, channels, height, width).

        Returns:
            torch.Tensor: The input image with added Gaussian noise.
        """
        return image + torch.randn(image.size()) * self.std + self.mean



# modified version from https://pytorch.org/tutorials/beginner/introyt/trainingyt.html
def train_one_epoch(model, data_loader, loss_fn, optimizer):
    """
    Trains a model for one epoch using the provided data loader, loss function, and optimizer.

    This function performs a single training epoch, iterating over the data loader to compute predictions,
    calculate the loss, backpropagate the gradients, and update the model parameters. Additionally, it
    computes the running loss and accuracy for reporting purposes.

    Args:
        model (torch.nn.Module): The model to be trained. Must implement a `forward` method.
        data_loader (torch.utils.data.DataLoader): DataLoader providing the training data in batches.
        loss_fn (callable): The loss function to compute the difference between predictions and labels.
        optimizer (torch.optim.Optimizer): The optimizer to update model parameters.

    Returns:
        tuple: A tuple containing:
            - float: The average loss over the entire dataset.
            - float: The average accuracy of the model over the entire dataset.
    """
    model.train(True)  # Set the model to training mode
    running_loss = 0.  # Initialize the cumulative loss
    accuracy = 0.      # Initialize the cumulative accuracy

    # Loop through each batch of data
    for i, data in enumerate(data_loader):
        # Each data batch contains inputs and corresponding labels
        inputs, labels = data

        # Zero the gradients before backpropagation
        optimizer.zero_grad()

        # Perform forward pass to get predictions
        outputs = model(inputs)

        # Compute the loss using the loss function
        loss = loss_fn(outputs[1], labels)  # Use the second output for loss computation
        loss.backward()  # Backpropagate the gradients

        # Update model parameters using the optimizer
        optimizer.step()

        # Accumulate the running loss
        running_loss += loss.item()

        # Calculate batch accuracy
        _, predicted = torch.max(outputs[1].data, 1)  # Get the predicted class
        accuracy += (predicted == labels).sum().item()  # Count correct predictions

    # Calculate average loss and accuracy over the dataset
    average_loss = running_loss / len(data_loader.dataset)
    average_accuracy = accuracy / len(data_loader.dataset)

    return average_loss, average_accuracy



def align_representation_mse(model_id, model_ood, data_loader_id, data_loader_ood, loss_fn, optimizer_ood, layer_index):
    """
    Aligns the representations of in-distribution and out-of-distribution models using Mean Squared Error (MSE).

    This function optimizes an out-of-distribution (OOD) model to align its intermediate feature representations
    with those of an in-distribution (ID) model for specified layers. It computes the alignment loss as the MSE
    between corresponding feature maps and optimizes the OOD model using backpropagation. Additionally, it computes
    the average alignment loss and classification accuracy over the dataset.

    Args:
        model_id (torch.nn.Module): The in-distribution model whose representations are used as the target.
        model_ood (torch.nn.Module): The out-of-distribution model being optimized.
        data_loader_id (torch.utils.data.DataLoader): DataLoader providing in-distribution data in batches.
        data_loader_ood (torch.utils.data.DataLoader): DataLoader providing out-of-distribution data in batches.
        loss_fn (callable): Loss function to compute the alignment loss (e.g., Mean Squared Error).
        optimizer_ood (torch.optim.Optimizer): Optimizer for updating the OOD model parameters.
        layer_index (list of int): Indices of layers whose feature maps are to be aligned.

    Returns:
        tuple: A tuple containing:
            - float: The average alignment loss over the entire OOD dataset.
            - float: The classification accuracy of the OOD model over the OOD dataset.
    """
    model_ood.train(True)  # Set the OOD model to training mode
    running_loss = 0.  # Initialize the cumulative alignment loss
    accuracy = 0.      # Initialize the cumulative accuracy

    # Loop through batches of ID and OOD data
    for data_id, data_ood in zip(data_loader_id, data_loader_ood):
        # Extract inputs and labels for ID and OOD data
        inputs_id, labels_id = data_id
        inputs_ood, labels_ood = data_ood

        # Zero the gradients for the OOD model
        optimizer_ood.zero_grad()

        # Perform forward passes for ID and OOD models
        outputs_id = model_id(inputs_id)  # Get ID model outputs
        outputs_ood = model_ood(inputs_ood)  # Get OOD model outputs

        # Compute alignment loss as the sum of MSE for specified layers
        align_loss = 0.0
        for layer in layer_index:
            align_loss += loss_fn(outputs_ood[0][layer], outputs_id[0][layer])

        # Backpropagate the alignment loss
        align_loss.backward()

        # Update the OOD model parameters
        optimizer_ood.step()

        # Accumulate the running alignment loss
        running_loss += align_loss.item()

        # Calculate classification accuracy for the OOD model
        _, predicted = torch.max(outputs_ood[1].data, 1)  # Get predicted classes
        accuracy += (predicted == labels_ood).sum().item()  # Count correct predictions

    # Calculate average alignment loss and accuracy over the OOD dataset
    average_loss = running_loss / len(data_loader_ood.dataset)
    average_accuracy = accuracy / len(data_loader_ood.dataset)

    return average_loss, average_accuracy




def align_representation_ot(model_id, model_ood, data_loader_id, data_loader_ood, optimizer_ood, weights, layer_index):
    """
    Aligns the representations of in-distribution and out-of-distribution models using Optimal Transport (OT).

    This function optimizes an out-of-distribution (OOD) model to align its intermediate feature representations
    with those of an in-distribution (ID) model for specified layers. The alignment is performed using Optimal
    Transport (OT) to compute the distance between feature distributions. Additionally, it computes the average
    alignment loss and classification accuracy over the dataset.

    Args:
        model_id (torch.nn.Module): The in-distribution model whose representations are used as the target.
        model_ood (torch.nn.Module): The out-of-distribution model being optimized.
        data_loader_id (torch.utils.data.DataLoader): DataLoader providing in-distribution data in batches.
        data_loader_ood (torch.utils.data.DataLoader): DataLoader providing out-of-distribution data in batches.
        optimizer_ood (torch.optim.Optimizer): Optimizer for updating the OOD model parameters.
        weights (torch.Tensor): Weight vector used for the Optimal Transport computation.
        layer_index (list of int): Indices of layers whose feature maps are to be aligned.

    Returns:
        tuple: A tuple containing:
            - float: The average alignment loss over the entire OOD dataset.
            - float: The classification accuracy of the OOD model over the OOD dataset.
    """
    model_ood.train(True)  # Set the OOD model to training mode
    running_loss = 0.  # Initialize the cumulative alignment loss
    accuracy = 0.      # Initialize the cumulative accuracy

    # Loop through batches of ID and OOD data
    for data_id, data_ood in zip(data_loader_id, data_loader_ood):
        # Extract inputs and labels for ID and OOD data
        inputs_id, labels_id = data_id
        inputs_ood, labels_ood = data_ood

        # Zero the gradients for the OOD model
        optimizer_ood.zero_grad()

        # Perform forward passes for ID and OOD models
        outputs_id = model_id(inputs_id)  # Get ID model outputs
        outputs_ood = model_ood(inputs_ood)  # Get OOD model outputs

        # Compute alignment loss using Optimal Transport
        align_loss = 0.0
        for layer in layer_index:
            x_id = torch.reshape(outputs_id[0][layer], (inputs_id.size(0), -1))  # Reshape ID features
            x_ood = torch.reshape(outputs_ood[0][layer], (inputs_ood.size(0), -1))  # Reshape OOD features
            M = ot.dist(x_ood, x_id)  # Compute the pairwise distance matrix
            align_loss += ot.emd2(weights, weights, M)  # Compute OT loss

        # Backpropagate the alignment loss
        align_loss.backward()

        # Update the OOD model parameters
        optimizer_ood.step()

        # Accumulate the running alignment loss
        running_loss += align_loss.item()

        # Calculate classification accuracy for the OOD model
        _, predicted = torch.max(outputs_ood[1].data, 1)  # Get predicted classes
        accuracy += (predicted == labels_ood).sum().item()  # Count correct predictions

    # Calculate average alignment loss and accuracy over the OOD dataset
    average_loss = running_loss / len(data_loader_ood.dataset)
    average_accuracy = accuracy / len(data_loader_ood.dataset)

    return average_loss, average_accuracy




def test_one_epoch(model, data_loader, loss_fn):
    """
    Evaluates the model for one epoch on the given dataset.

    This function calculates the average loss and accuracy of a model
    on a given dataset using a specified loss function. The model is
    set to evaluation mode to ensure deterministic behavior.

    Args:
        model (torch.nn.Module): The model to be evaluated.
        data_loader (torch.utils.data.DataLoader): DataLoader providing the test data in batches.
        loss_fn (torch.nn.Module): Loss function used to compute the loss.

    Returns:
        tuple: A tuple containing:
            - float: The average loss over the entire dataset.
            - float: The accuracy of the model over the entire dataset.
    """
    # Set the model to evaluation mode, disabling dropout and using population statistics for batch normalization
    model.eval()

    accuracy = 0.0  # Initialize the cumulative accuracy
    running_loss = 0.0  # Initialize the cumulative loss

    # Disable gradient computation to reduce memory consumption during evaluation
    with torch.no_grad():
        # Iterate through the DataLoader
        for data in data_loader:
            inputs, labels = data  # Extract inputs and labels
            outputs = model(inputs)  # Perform a forward pass

            # Calculate predictions and update accuracy
            _, predictions = torch.max(outputs[1].data, 1)  # Get predicted classes
            accuracy += (predictions == labels).sum().item()  # Count correct predictions

            # Compute and accumulate loss
            running_loss += loss_fn(outputs[1], labels).item()

    # Calculate average loss and accuracy over the dataset
    average_loss = running_loss / len(data_loader.dataset)
    average_accuracy = accuracy / len(data_loader.dataset)

    return average_loss, average_accuracy

