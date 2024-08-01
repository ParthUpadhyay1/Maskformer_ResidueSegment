
# Import the necessary packages
import random
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, random_split
import albumentations as A
import numpy as np
import pandas as pd
from datasets import load_dataset
from transformers import (
    MaskFormerConfig,
    MaskFormerImageProcessor,
    MaskFormerModel,
    MaskFormerForInstanceSegmentation,
)
import evaluate
from huggingface_hub import notebook_login

import wandb
import os
from PIL import Image
import torchvision.transforms as transforms

# Login to Weights & Biases
wandb.login()

# Initialize a new W&B run
wandb.init(project='Residue-Segmentation-1', entity='agrifarm')

# # Set up configuration
# config = wandb.config
# config.learning_rate = 0.001
# config.batch_size = 16
# config.epochs = 50
# config.buffer_size = 10000

# Create the dataset
# image_path = 'U:\\a_ImageData_Draft_moved_DNU\\CoverCrops\\All_2018_2019_1m_Images\\BigDataset\\Images\\'
# mask_path = 'U:\\a_ImageData_Draft_moved_DNU\\CoverCrops\\All_2018_2019_1m_Images\\BigDataset\\Masks\\'

image_path = 'Images/'
mask_path = 'Masks/'

# Create the MaskFormer Image Preprocessor
processor = MaskFormerImageProcessor(
    reduce_labels=True,
    size=(256, 256),
    ignore_index=255,
    do_resize=False,
    do_rescale=False,
    do_normalize=False,
)

# Define the name of the model
model_name = "facebook/maskformer-swin-base-ade"
# Get the MaskFormer config and print it
config = MaskFormerConfig.from_pretrained(model_name)

# Use the config object to initialize a MaskFormer model with randomized weights
model = MaskFormerForInstanceSegmentation(config)
# Replace the randomly initialized model with the pre-trained model weights
base_model = MaskFormerModel.from_pretrained(model_name)
model.model = base_model

# Define the configurations of the transforms specific
# to the dataset used
ADE_MEAN = np.array([123.675, 116.280, 103.530]) / 255.0
ADE_STD = np.array([58.395, 57.120, 57.375]) / 255.0
ADE_MEAN = ADE_MEAN.tolist()
ADE_STD = ADE_STD.tolist()
# Build the augmentation transforms
train_val_transform = A.Compose([
    A.Resize(width=256, height=256),
    A.HorizontalFlip(p=0.3),
    A.Normalize(mean=ADE_MEAN, std=ADE_STD),
])

class ImageSegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, processor, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        # self.image_filenames = os.listdir(image_dir)
        self.image_filenames = [f for f in os.listdir(image_dir) if f.lower().endswith('.png')]
        self.transform = transform
        self.processor = processor
    
    def __len__(self):
        return len(self.image_filenames)
    
    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.image_filenames[idx])
        mask_name = os.path.join(self.mask_dir, self.image_filenames[idx])
        
        image = Image.open(img_name).convert('RGB')
        image = np.array(image)
        # mask = Image.open(mask_name).convert('L')
        
        # if self.transform:
        #     image = self.transform(image)
        #     mask = self.transform(mask)

        
        # of shape (height, width)
        instance_seg = np.array(Image.open(mask_name))[..., 1]
        class_id_map = np.array(Image.open(mask_name))[..., 0]
        class_labels = np.unique(class_id_map)
        # print (class_labels)

        
        # Build the instance to class dictionary
        inst2class = {}
        for label in class_labels:
            instance_ids = np.unique(instance_seg[class_id_map == label])
            inst2class.update({i: label for i in instance_ids})
            
        # Apply transforms
        if self.transform is not None:
            transformed = self.transform(image=image, mask=instance_seg)
            # image = transformed(image)
            # instance_seg = transformed(instance_seg)
            
            # transformed = self.transform(image=image, mask=instance_seg)
            (image, instance_seg) = (transformed["image"], transformed["mask"])
            
            # Convert from channels last to channels first
            image = image.transpose(2,0,1)
        
            # instance_id_to_semantic_id=inst2class,

        
        inputs = self.processor(
            [image],
            [instance_seg],
            return_tensors="pt"
        )
        inputs = {
            k:v.squeeze() if isinstance(v, torch.Tensor) else v[0] for k,v in inputs.items()
        }
        # Return the inputs
        return inputs

# Creating image dataset
dataset = ImageSegmentationDataset(image_dir=image_path, mask_dir=mask_path, processor=processor, transform=train_val_transform)

# Split the dataset into training and validation sets
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

def collate_fn(examples):
    # Get the pixel values, pixel mask, mask labels, and class labels
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    pixel_mask = torch.stack([example["pixel_mask"] for example in examples])
    mask_labels = [example["mask_labels"] for example in examples]
    class_labels = [example["class_labels"] for example in examples]
    # Return a dictionary of all the collated features
    return {
        "pixel_values": pixel_values,
        "pixel_mask": pixel_mask,
        "mask_labels": mask_labels,
        "class_labels": class_labels
    }
# Building the training and validation dataloader
train_dataloader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True,
    collate_fn=collate_fn
)
val_dataloader = DataLoader(
    val_dataset,
    batch_size=16,
    shuffle=False,
    collate_fn=collate_fn
)

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
# Initialize Adam optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
# Set number of epochs and batch size
num_epochs = 25
for epoch in range(num_epochs):
    print(f"Epoch {epoch} | Training")
    # Set model in training mode 
    model.train()
    train_loss, val_loss = [], []
    # Training loop
    for idx, batch in enumerate(tqdm(train_dataloader)):
        # Reset the parameter gradients
        optimizer.zero_grad()
 
        # Forward pass
        outputs = model(
            pixel_values=batch["pixel_values"].to(device),
            mask_labels=[labels.to(device) for labels in batch["mask_labels"]],
            class_labels=[labels.to(device) for labels in batch["class_labels"]],
        )
        # Backward propagation
        loss = outputs.loss
        train_loss.append(loss.item())
        loss.backward()
        if idx % 50 == 0:
            print("  Training loss: ", round(sum(train_loss)/len(train_loss), 6))
        # Optimization
        optimizer.step()
    # Average train epoch loss
    train_loss = sum(train_loss)/len(train_loss)
    # Set model in evaluation mode
    model.eval()
    start_idx = 0
    print(f"Epoch {epoch} | Validation")
    for idx, batch in enumerate(tqdm(val_dataloader)):
        with torch.no_grad():
            # Forward pass
            outputs = model(
                pixel_values=batch["pixel_values"].to(device),
                mask_labels=[labels.to(device) for labels in batch["mask_labels"]],
                class_labels=[labels.to(device) for labels in batch["class_labels"]],
            )
            # Get validation loss
            loss = outputs.loss
            val_loss.append(loss.item())
            if idx % 50 == 0:
                print("  Validation loss: ", round(sum(val_loss)/len(val_loss), 6))
    # Average validation epoch loss
    val_loss = sum(val_loss)/len(val_loss)
    # Print epoch losses
    print(f"Epoch {epoch} | train_loss: {train_loss} | validation_loss: {val_loss}")

# Saving trained model
modelPath = "MaskformerModel/"
model.save_pretrained(modelPath)
processor.save_pretrained(modelPath)
