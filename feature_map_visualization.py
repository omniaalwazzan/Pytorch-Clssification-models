import torch
import matplotlib.pyplot as plt
from torchvision import models
import torch.nn as nn
from PIL import Image
import numpy as np

from torchsummary import summary

#%%
class convNext(nn.Module):
    def __init__(self):
        super(convNext, self).__init__()
        model = models.convnext_base(weights=models.ConvNeXt_Base_Weights.IMAGENET1K_V1)
        self.feature_extractor = nn.Sequential(*list(model.children())[:-1])

    def forward(self, x):
        feature = self.feature_extractor(x)
        return feature
model = convNext()
summary(model, (3,128,128))
#%%
image_patch = r"C:\Users\Omnia\Desktop\data\NH06-876\NH06-876 d-4.3385_x-15519_y-19953_w-2221_h-2221.tif"
image_path = r"C:\Users\Omnia\Pictures\dog.jpg"

image = Image.open(image_patch)
image_np = np.array(image)

# Preprocess the image (you might need to adjust this based on your actual preprocessing)
preprocessed_image = torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0).float()

# Initialize the model
model = convNext()
# Forward pass through the model
feature_output = model(preprocessed_image) # torch.Size([1, 1024, 1, 1])
print('feature-out:',feature_output.shape)
#%%
#The output is of shape [batch_size, channels, height, width])
reshaped_feature = feature_output.squeeze().detach().numpy() # (1024,)
# Create an array of indices for the x-axis
indices = np.arange(len(reshaped_feature))
#%%
# This if we extracted last featuer vector coming from pooling
# Plot the feature map using a bar plot, for extracting model.children())[:-1]
plt.bar(indices, reshaped_feature, color='blue')
plt.title("Feature Map")
plt.xlabel("Channel Index")
plt.ylabel("Activation Value")
plt.show()

#%%
# Get the three channels
channel1 = reshaped_feature[ 0, :, :]
channel2 = reshaped_feature[ 1, :, :]
channel3 = reshaped_feature[ 2, :, :]

# Plot each channel as a separate subfigure
# This if we extracted last featuer vector before pooling :feature-out: torch.Size([1, 1024, 16, 16])
# for extracting model.children())[:-2]
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(channel1, cmap='viridis')
plt.title('Channel 1')

plt.subplot(1, 3, 2)
plt.imshow(channel2, cmap='viridis')
plt.title('Channel 2')

plt.subplot(1, 3, 3)
plt.imshow(channel3, cmap='viridis')
plt.title('Channel 3')

plt.show()
