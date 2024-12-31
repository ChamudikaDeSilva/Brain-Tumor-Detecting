import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader
from torchvision.transforms import Resize
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import random
import cv2
from torch.autograd import Function

# Define the custom dataset class
class BrainTumorDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.labels = []
        self.classes = ['glioma', 'meningioma', 'notumor', 'pituitary']
        
        # Load images and labels
        for label in self.classes:
            class_dir = os.path.join(root_dir, label)
            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)
                self.images.append(img_path)
                self.labels.append(self.classes.index(label))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        label = self.labels[idx]
        return image, label

# Define transformations
transform = transforms.Compose([
    Resize((224, 224)),  # Resize images to the size expected by MobileNet
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),  # MobileNet normalization
])

# Create datasets and data loaders
train_dataset = BrainTumorDataset(root_dir='C:/laragon/www/Brain-Tumor-Detection/Training', transform=transform)
val_dataset = BrainTumorDataset(root_dir='C:/laragon/www/Brain-Tumor-Detection/Testing', transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Load the pre-trained MobileNetV2 model and modify the final layer
model = models.mobilenet_v2(weights=True)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(train_dataset.classes))

# Move the model to the GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop parameters
num_epochs = 150
train_acc_history = []
val_acc_history = []
train_loss_history = []
val_loss_history = []
best_val_loss = float('inf')
early_stop_counter = 0
early_stop_patience = 10  # Stop training after 10 epochs without improvement

# Define the checkpoint function
def save_checkpoint(state, filename="best_model.pth"):
    torch.save(state, filename)

# Training loop
for epoch in range(num_epochs):
    model.train()
    running_loss_train = 0.0
    correct_train = 0
    total_train = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)  # Outputs shape: (batch_size, num_classes)

        loss = criterion(outputs, labels)

        loss.backward()
        
        optimizer.step()

        running_loss_train += loss.item()

        _, predicted_train = torch.max(outputs, dim=1)

        total_train += labels.size(0)
        correct_train += (predicted_train == labels).sum().item()

    train_loss_epoch = running_loss_train / len(train_loader)
    train_acc_epoch = correct_train / total_train * 100

    train_loss_history.append(train_loss_epoch)
    train_acc_history.append(train_acc_epoch)

    model.eval()
    running_loss_val = 0.0
    correct_val = 0
    total_val = 0
    
    y_true_val = []
    y_pred_val = []

    with torch.no_grad():
        for inputs_val, labels_val in val_loader:
            inputs_val, labels_val = inputs_val.to(device), labels_val.to(device)

            outputs_val = model(inputs_val)

            loss_val = criterion(outputs_val, labels_val)

            running_loss_val += loss_val.item()

            _, predicted_val = torch.max(outputs_val, dim=1)

            total_val += labels_val.size(0)
            correct_val += (predicted_val == labels_val).sum().item()

            y_true_val.extend(labels_val.cpu().numpy())
            y_pred_val.extend(predicted_val.cpu().numpy())

    val_loss_epoch = running_loss_val / len(val_loader)
    val_acc_epoch = correct_val / total_val * 100

    val_loss_history.append(val_loss_epoch)
    val_acc_history.append(val_acc_epoch)

    # Check for early stopping and model checkpointing
    if val_loss_epoch < best_val_loss:
        best_val_loss = val_loss_epoch
        save_checkpoint({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': val_loss_epoch,
        })
        early_stop_counter = 0
    else:
        early_stop_counter += 1

    if early_stop_counter >= early_stop_patience:
        print(f"Early stopping at epoch {epoch + 1}")
        break

    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss_epoch:.4f}, Train Acc: {train_acc_epoch:.2f}%, Val Loss: {val_loss_epoch:.4f}, Val Acc: {val_acc_epoch:.2f}%')

# Plot accuracy and loss graphs
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(train_acc_history, label='Train Accuracy')
plt.plot(val_acc_history, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_loss_history, label='Train Loss')
plt.plot(val_loss_history, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss')
plt.legend()
plt.tight_layout()
plt.show()

# Confusion Matrix
cm = confusion_matrix(y_true_val, y_pred_val)
plt.figure(figsize=(8, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=train_dataset.classes, yticklabels=train_dataset.classes)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()


####################### GRAD-CAM IMPLEMENTATIONS #######################
# Define the custom dataset class
class BrainTumorDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.labels = []
        self.classes = ['glioma', 'meningioma', 'notumor', 'pituitary']
        
        # Load images and labels
        for label in self.classes:
            class_dir = os.path.join(root_dir, label)
            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)
                self.images.append(img_path)
                self.labels.append(self.classes.index(label))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        label = self.labels[idx]
        return image, label

# Grad-CAM helper functions
def get_gradcam_heatmap(model, img_tensor, target_class):
    model.eval()
    conv_layer = model.features[-1]  # Last convolutional layer of the model
    
    gradients = []
    
    # Hook to capture gradients
    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])  # Save the gradients
    
    # Hook to capture forward activations (feature maps)
    def forward_hook(module, input, output):
        model.feature_maps = output
    
    # Register hooks
    handle_backward = conv_layer.register_backward_hook(backward_hook)
    handle_forward = conv_layer.register_forward_hook(forward_hook)

    # Perform forward pass
    output = model(img_tensor.unsqueeze(0).to(device))
    model.zero_grad()
    
    # Perform backward pass for the target class
    output[:, target_class].backward()
    
    # Get the gradients and feature maps
    gradients = gradients[0].cpu().data.numpy()[0]
    feature_maps = model.feature_maps.cpu().data.numpy()[0]
    
    # Calculate weights as the average of the gradients
    weights = np.mean(gradients, axis=(1, 2))
    
    # Compute the weighted sum of the feature maps
    cam = np.zeros(feature_maps.shape[1:], dtype=np.float32)
    for i, w in enumerate(weights):
        cam += w * feature_maps[i]

    # Apply ReLU to remove negative values
    cam = np.maximum(cam, 0)
    
    # Normalize the heatmap
    cam = cv2.resize(cam, (img_tensor.shape[2], img_tensor.shape[1]))
    heatmap = cam - np.min(cam)
    heatmap = heatmap / (np.max(heatmap) + 1e-8)

    # Remove the hooks
    handle_forward.remove()
    handle_backward.remove()
    
    return heatmap

# Guided Backpropagation ReLU Function
class GuidedBackpropReLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        positive_mask = (input > 0).type_as(input)
        ctx.save_for_backward(positive_mask)
        return input.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        positive_mask, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[positive_mask == 0] = 0
        return grad_input

# Replace ReLU with GuidedBackpropReLU in the model
for module in model.modules():
    if isinstance(module, nn.ReLU):
        module.register_forward_hook(lambda module, grad_in, grad_out: GuidedBackpropReLU.apply(grad_in[0]))

# Guided backpropagation function
def guided_backpropagation(model, img_tensor):
    img_tensor.requires_grad = True
    model.eval()
    
    output = model(img_tensor.unsqueeze(0).to(device))
    target_class = output.argmax(dim=1).item()

    model.zero_grad()
    output[:, target_class].backward()

    # Ensure gradients exist
    if img_tensor.grad is None:
        raise ValueError("Gradients are None. Ensure that `requires_grad=True` is set for the input tensor.")
    
    grad = img_tensor.grad.cpu().data.numpy()
    
    if grad.ndim != 3:
        raise ValueError(f"Expected gradient to have 3 dimensions [C, H, W], but got {grad.shape}.")
    
    return grad[0]

# Overlay Guided Grad-CAM
def overlay_guided_gradcam(model, img, target_class):
    gradcam_heatmap = get_gradcam_heatmap(model, img, target_class)
    guided_grads = guided_backpropagation(model, img)
    
    print(f"Grad-CAM heatmap shape: {gradcam_heatmap.shape}")
    print(f"Guided gradients shape: {guided_grads.shape}")

    # Ensure guided_grads is a 3D array by adding a channel dimension if needed
    if guided_grads.ndim == 2:
        guided_grads = np.expand_dims(guided_grads, axis=0)
    
    # Resize the Grad-CAM heatmap to match the spatial dimensions of guided gradients
    gradcam_heatmap_resized = cv2.resize(gradcam_heatmap, (guided_grads.shape[2], guided_grads.shape[1]))

    print(f"Resized Grad-CAM heatmap shape: {gradcam_heatmap_resized.shape}")
    
    # Normalize the Grad-CAM heatmap and guided gradients
    gradcam_heatmap_resized = np.expand_dims(gradcam_heatmap_resized, axis=0)  # Add a channel dimension
    gradcam_heatmap_resized = gradcam_heatmap_resized / np.max(gradcam_heatmap_resized)  # Normalize
    
    # Multiply guided gradients with resized Grad-CAM heatmap
    guided_gradcam = gradcam_heatmap_resized * guided_grads
    
    # Squeeze the channel dimension
    guided_gradcam = np.squeeze(guided_gradcam, axis=0)
    
    return np.maximum(guided_gradcam, 0)

# Occlusion Sensitivity Map Implementation
def occlusion_sensitivity_map(model, img_tensor, patch_size=16, stride=8):
    model.eval()
    original_output = model(img_tensor.unsqueeze(0).to(device))
    target_class = original_output.argmax(dim=1).item()

    sensitivity_map = np.zeros((img_tensor.shape[1], img_tensor.shape[2]))

    for i in range(0, img_tensor.shape[1] - patch_size + 1, stride):
        for j in range(0, img_tensor.shape[2] - patch_size + 1, stride):
            occluded_img = img_tensor.clone()
            occluded_img[:, i:i + patch_size, j:j + patch_size] = 0  # Apply patch occlusion

            output = model(occluded_img.unsqueeze(0).to(device))
            occlusion_score = output[0, target_class].item()

            sensitivity_map[i:i + patch_size, j:j + patch_size] = original_output[0, target_class].item() - occlusion_score

    return sensitivity_map

# Superimposed Activation Mask
def superimposed_activation_mask(model, img_tensor, target_class, threshold=0.5):
    gradcam_heatmap = get_gradcam_heatmap(model, img_tensor, target_class)

    activation_mask = gradcam_heatmap > threshold
    mask = activation_mask.astype(np.float32)

    # Detach the tensor before converting to a NumPy array
    img = img_tensor.permute(1, 2, 0).detach().cpu().numpy()
    img = (img - np.min(img)) / (np.max(img) - np.min(img))  # Normalize image
    img = (img * 255).astype(np.uint8)  # Convert to uint8
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # Normalize and convert Grad-CAM heatmap to uint8
    gradcam_heatmap = (gradcam_heatmap * 255).astype(np.uint8)
    heatmap_colored = cv2.applyColorMap(gradcam_heatmap, cv2.COLORMAP_JET)

    # Ensure img and heatmap_colored are both uint8
    if img.dtype != np.uint8:
        img = img.astype(np.uint8)
    if heatmap_colored.dtype != np.uint8:
        heatmap_colored = heatmap_colored.astype(np.uint8)

    # Overlay the heatmap on the image
    overlay = cv2.addWeighted(img, 0.6, heatmap_colored, 0.4, 0)

    return overlay


# Plot Grad-CAM, Guided Grad-CAM, Occlusion Sensitivity, and Superimposed Activation Mask for 5 random images
fig, axs = plt.subplots(5, 4, figsize=(20, 25))
axs = axs.ravel()

for i in range(5):
    idx = random.randint(0, len(val_dataset) - 1)
    img, label = val_dataset[idx]

    output = model(img.unsqueeze(0).to(device))
    pred_class = output.argmax(dim=1).item()

    heatmap = get_gradcam_heatmap(model, img, pred_class)
    guided_gradcam = overlay_guided_gradcam(model, img, pred_class)
    occlusion_map = occlusion_sensitivity_map(model, img)
    activation_mask = superimposed_activation_mask(model, img, pred_class)

    # Detach the image tensor before converting it to a NumPy array
    axs[i * 4].imshow(img.permute(1, 2, 0).detach().cpu().numpy())
    axs[i * 4].set_title(f"Original (Label: {val_dataset.classes[label]})")
    
    axs[i * 4 + 1].imshow(heatmap, cmap='jet')
    axs[i * 4 + 1].set_title("Grad-CAM Heatmap")

    axs[i * 4 + 2].imshow(guided_gradcam, cmap='jet')
    axs[i * 4 + 2].set_title("Guided Grad-CAM")

    axs[i * 4 + 3].imshow(occlusion_map, cmap='viridis')
    axs[i * 4 + 3].set_title("Occlusion Sensitivity")

plt.tight_layout()
plt.show()