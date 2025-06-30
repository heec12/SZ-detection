import numpy as np
import sys
import torch
import torch.nn as nn
import random
import matplotlib.pyplot as plt

# SE Block for attention mechanism
class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SEBlock, self).__init__()
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1)
        self.fc2 = nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1)

    def forward(self, x):
        scale = torch.mean(x, dim=(2, 3), keepdim=True)
        scale = torch.relu(self.fc1(scale))
        scale = torch.sigmoid(self.fc2(scale))
        return x * scale

# Define the focal loss class
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)  # Prevents NaNs when probability is 0
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        return F_loss.mean()

# Define the FCN model with improvements
class FCN(nn.Module):
    def __init__(self):
        super(FCN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.se1 = SEBlock(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.se2 = SEBlock(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.se3 = SEBlock(128)
        self.conv4 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.se4 = SEBlock(64)
        self.conv5 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(32)
        self.se5 = SEBlock(32)
        self.conv6 = nn.Conv2d(32, 1, kernel_size=1)

        # Define pooling, upsampling, and dropout layers
        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        self.dropout = nn.Dropout(0.75)

    def forward(self, x):
        x = self.bn1(self.conv1(x))
        x = self.se1(x)
        x = self.maxpool(x)
        x = self.dropout(self.bn2(self.conv2(x)))
        x = self.se2(x)
        x = self.maxpool(x)
        x = self.dropout(self.bn3(self.conv3(x)))
        x = self.se3(x)
        x = self.upsample(x)
        x = self.bn4(self.conv4(x))
        x = self.se4(x)
        x = self.upsample(x)
        x = self.bn5(self.conv5(x))
        x = self.se5(x)
        x = self.conv6(x)
        return x

# Function for manual data augmentation
def augment_data(x, y):
    if random.random() > 0.5:
        x = torch.flip(x, [2])  # Horizontal flip
        y = torch.flip(y, [0])
    if random.random() > 0.5:
        x = torch.flip(x, [1])  # Vertical flip
        y = torch.flip(y, [1])
    if random.random() > 0.5:
        angle = random.randint(-10, 10)  # Random rotation between -10 and 10 degrees
        x = rotate_tensor(x, angle)
        y = rotate_tensor(y.unsqueeze(0), angle).squeeze(0)
    return x, y

def rotate_tensor(tensor, angle):
    angle = np.deg2rad(angle)

    # Create the rotation matrix on the same device as the tensor
    rotation_matrix = torch.tensor([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle), np.cos(angle), 0],
        [0, 0, 1]
    ], device=tensor.device)  # Ensuring rotation_matrix is on the same device as tensor

    # Create the affine grid on the same device
    grid = nn.functional.affine_grid(rotation_matrix.unsqueeze(0)[:, :2, :].float(), tensor.unsqueeze(0).size(),
                                     align_corners=False)

    # Apply grid_sample, ensuring both tensor and grid are on the same device
    tensor = nn.functional.grid_sample(tensor.unsqueeze(0), grid, align_corners=False)

    return tensor.squeeze(0)

def calculate_class_weights(y_train_all):
    # Flatten the labels to count occurrences
    labels_flat = torch.cat([y.flatten() for y in y_train_all])
    # Count occurrences of each class (0 = background, 1 = subduction zone)
    class_counts = torch.bincount(labels_flat.long())
    # Calculate weights for each class
    class_weights = 1.0 / class_counts.float()
    # Normalize so that the sum of the weights is 1
    class_weights = class_weights / class_weights.sum()

    return class_weights

# Dice Loss function
def dice_loss(pred, target, smooth=1.0):
    pred = torch.sigmoid(pred)
    intersection = (pred * target).sum()
    return 1 - ((2.0 * intersection + smooth) / (pred.sum() + target.sum() + smooth))

def combined_loss(pred, target, class_weights, focal_alpha=1, focal_gamma=4):
    # Class balancing for BCEWithLogitsLoss
    bce_loss = nn.BCEWithLogitsLoss(pos_weight=class_weights[1])(pred,target)  # Weight positive (subduction zone) class
    # Dice Loss
    d_loss = dice_loss(pred, target)
    # Focal Loss
    f_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)(pred, target)
    # Combine the losses
    return bce_loss + d_loss + f_loss

# Function to train the model
def train_model(model, criterion, optimizer, scheduler, X_train, y_train, device, epochs=8):
    model.train()  # Set the model to training mode
    for epoch in range(epochs):
        running_loss = 0.0
        for i in range(len(X_train)):
            x, y = augment_data(X_train[i], y_train[i])  # Apply augmentation
            x = x.to(device).float()
            y = y.type(torch.FloatTensor).to(device)

            optimizer.zero_grad()
            outputs = model(x.unsqueeze(0))
            outputs = outputs.squeeze()  # To match the dimension
            loss = criterion(outputs, y)  # Use the updated combined loss
            loss.backward()

            # Clip gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item()

        scheduler.step()  # Adjust the learning rate
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(X_train):.4f}")

# Function to evaluate the model
def evaluate_model(model, criterion, X_eval, y_eval, device):
    model.eval()  # Set the model to evaluation mode
    total_loss = 0.0
    with torch.no_grad():
        for x, y in zip(X_eval, y_eval):
            x = x.to(device).float()
            y = y.to(device).float()

            outputs = model(x.unsqueeze(0))
            outputs = outputs.squeeze()  # To match the dimension
            loss = criterion(outputs, y)
            total_loss += loss.item()

    average_loss = total_loss / len(X_eval)
    return average_loss, outputs

# Post-processing function
def post_process(outputs, threshold=0.45):
    outputs = torch.sigmoid(outputs)
    outputs = (outputs > threshold).float()
    return outputs

# Function to create a predicted SZ
def plot_results(y_matrix, title):
    plt.figure(figsize=(8, 2))
    plt.imshow(y_matrix, vmin=0, cmap='binary')
    plt.title(title)
    plt.colorbar(label='Boolean Value')
    plt.tight_layout()
    plt.show()

# Function to create the original RGB image
def plot_rgb(loaded_tensor_rgb, title):
    np_array = loaded_tensor_rgb.cpu().numpy()
    plt.figure(figsize=(8, 2))
    plt.title(title)
    plt.imshow(np_array)
    plt.show()

# Function to create a predicted SZ
def plot_actual(y_matrix, title):
    y_matrix = y_matrix.squeeze(0)
    plt.figure(figsize=(8, 2))
    plt.imshow(y_matrix, vmin=0, cmap='binary')
    plt.title(title)
    plt.colorbar(label='Boolean Value')
    plt.tight_layout()
    plt.show()

def plot_combined_vertically(results, rgb_image, title, top_row):
    # Create subplots with specific height ratios: 1 for the first panel, 2 for the others
    fig, axs = plt.subplots(3, 1, figsize=(6, 9), gridspec_kw={'height_ratios': [1, 2, 2]})

    # Plot the first panel: top_row vs. x_values (smaller height)
    top_row_np = top_row.numpy()  # Convert top_row to NumPy array
    x_values = np.arange(top_row_np.shape[1])  # Generate x-axis values
    axs[0].plot(x_values, top_row_np[0, :])  # Plot the first row
    axs[0].set_xlim([min(x_values), max(x_values)])
    axs[0].tick_params(which='both', direction='in', top=True, right=True)  # Add ticks on all sides

    # Plot the predicted subduction zone
    im1 = axs[1].imshow(results, vmin=0, cmap='binary')
    axs[1].set_title('Predicted SZ')
    axs[1].tick_params(which='both', direction='in', top=True, right=True)  # Add ticks on all sides
    axs[1].grid(False)  # Remove gridlines, but keep the border
    cbar1 = fig.colorbar(im1, ax=axs[1], fraction=0.03, pad=0.04, shrink=0.5)  # Shorter colorbar

    # Plot the RGB image
    np_array = np.clip(rgb_image.cpu().numpy().transpose(1, 2, 0), 0, 255).astype(np.uint8)
    axs[2].imshow(np_array)
    axs[2].set_title('RGB Image')
    axs[2].tick_params(which='both', direction='in', top=True, right=True)  # Add ticks on all sides
    axs[2].grid(False)

    # Set a title for the entire plot
    plt.suptitle(title, fontsize=16)

    # Adjust layout for better spacing
    plt.tight_layout()
    plt.show()

# def train_and_evaluate_fcn(ha_temp, model, criterion, optimizer, scheduler, device):
#     # Load training data
#     input_temp = 'temp' + str(ha_temp)
#     input_temp_ha = 'ha_temp' + str(ha_temp)
#
#     # Define file paths for RGB and boolean images
#     filename_rgb = f'/rubin/s1/hxc5400/ml_connect_conv_py_code/rgb_images/{input_temp_ha}.pt'
#     filename_bool = f'/rubin/s1/hxc5400/ml_connect_conv_py_code/boolean_images/boolean_{input_temp}.o'
#
#     # Load training tensors
#     try:
#         X_train = torch.load(filename_rgb).to(device)
#         y_train = torch.load(filename_bool).to(device)
#     except FileNotFoundError as e:
#         print(f"File not found: {e}")
#         return
#
#     # Preprocess training data
#     X_train = X_train.float().permute(2, 0, 1).unsqueeze(0)  # Reshape for model
#     y_train = y_train.type(torch.FloatTensor).unsqueeze(0)    # Reshape labels
#
#     # Training loop
#     model.train()
#     optimizer.zero_grad()
#     outputs = model(X_train)
#     loss = criterion(outputs, y_train)
#     loss.backward()
#     optimizer.step()
#     scheduler.step()
#
#     # Print the loss
#     print(f"HA Temp {ha_temp}: Training Loss = {loss.item():.4f}")
#
#     # Evaluation loop
#     model.eval()
#     with torch.no_grad():
#         outputs_eval = model(X_train)
#         eval_loss = criterion(outputs_eval, y_train)
#
#     # Print the evaluation loss
#     print(f"HA Temp {ha_temp}: Evaluation Loss = {eval_loss.item():.4f}")

def apply_fcn(ha_temp, model, model_path, device):
    # Load data (input)
    input_temp_ha = 'rgb_ha_temp' + "{:0>4d}".format(int(ha_temp))
    filename_rgb = f'{model_path}{input_temp_ha}.pt'
    # print(f"Loading image: {filename_rgb}")

    # Load the input tensor (Assuming it was saved with torch.save)
    X_eval = torch.load(filename_rgb, weights_only=True).to(device)

    # Preprocess input data
    X_eval = X_eval.float().permute(2, 0, 1).unsqueeze(0)  # Reshape for model input

    # Apply the model
    model.eval()  # Make sure the model is in evaluation mode
    with torch.no_grad():  # Disable gradient calculations
        outputs = model(X_eval)
        outputs = torch.sigmoid(outputs)  # Apply sigmoid activation

        # Apply thresholding for binary classification
        outputs = (outputs >= 0.5).float()
        top_row = outputs.squeeze(0).squeeze(0)[0, :].unsqueeze(0)

        # # Debug raw and processed outputs
        # print("Binary outputs:", outputs.squeeze(0).squeeze(0).cpu().numpy())
        # print("Post-processed top_row:", top_row[0].cpu().numpy())

        # Post-process the top_row to detect edges
        rising_edges = np.where(np.diff(top_row[0].cpu().numpy()) == 1)[0] + 1
        falling_edges = np.where(np.diff(top_row[0].cpu().numpy()) == -1)[0] + 1

        # # Debug edge detection
        # print("Rising edges:", rising_edges)
        # print("Falling edges:", falling_edges)

        # Handle empty edge cases
        if len(rising_edges) == 0 or len(falling_edges) == 0:
            print(f"No edges detected for ha_temp {ha_temp}")
            return top_row, [], [], []

        # Find middle points between corresponding rising and falling edges
        mid_points = []
        for rise, fall in zip(rising_edges, falling_edges):
            mid_point = (rise + fall) // 2  # Integer midpoint between rise and fall
            mid_points.append(mid_point)

        # # Debug mid-points
        # print("Mid points:", mid_points)

        # Visualize the results
        # plot_combined_vertically(
        #     results=outputs.squeeze().cpu().numpy(),  # Ensure it's on the CPU
        #     rgb_image=X_eval.squeeze().cpu(),  # Ensure it's on the CPU
        #     title=f"Visualization for HA Temp {ha_temp}",
        #     top_row=top_row.cpu()  # Ensure it's on the CPU
        # )

        return top_row, rising_edges, falling_edges, mid_points

# def apply_fcn(ha_temp, model, model_path, device):
#     # Load data (input)
#     input_temp = 'temp' + str(ha_temp)
#     #input_temp_ha = 'rgb_ha_temp' + str("{:0>4d}".format(ha_temp))
#     input_temp_ha = 'rgb_ha_temp' + "{:0>4d}".format(int(ha_temp))
#
#     # Define file paths for RGB images
#     filename_rgb = f'{model_path}{input_temp_ha}.pt'
#
#     # Load the input tensor (Assuming it was saved with torch.save)
#     X_eval = torch.load(filename_rgb, weights_only=True).to(device)
#
#     # Preprocess input data
#     X_eval = X_eval.float().permute(2, 0, 1).unsqueeze(0)  # Reshape for model input
#
#     # Apply the model
#     model.eval()  # Make sure the model is in evaluation mode
#     with torch.no_grad():  # Disable gradient calculations
#         outputs = model(X_eval)
#         outputs = torch.sigmoid(outputs)  # Assuming sigmoid activation for output
#         top_row = outputs.squeeze(0).squeeze(0)[0, :].unsqueeze(0)
#
#         # Post-process the top_row to detect edges
#         rising_edges = np.where(np.diff(top_row.cpu().numpy()) == 1)[0] + 1
#         falling_edges = np.where(np.diff(top_row.cpu().numpy()) == -1)[0] + 1
#
#         # Find middle points between corresponding rising and falling edges
#         mid_points = []
#         for rise, fall in zip(rising_edges, falling_edges):
#             mid_point = (rise + fall) // 2  # Integer midpoint between rise and fall
#             mid_points.append(mid_point)
#
#         return top_row, rising_edges, falling_edges, mid_points
