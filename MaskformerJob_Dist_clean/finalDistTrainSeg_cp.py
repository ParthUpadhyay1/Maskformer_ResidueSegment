# Import the necessary packages
import random
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split, DistributedSampler
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
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# Login to Weights & Biases
wandb.login()

# Initialize a new W&B run
wandb.init(
    project='Distributed-Maskformer-1', 
    entity='agrifarm',
    name=f"experiment_1",
      # Track hyperparameters and run metadata
      config={
      "batch_size": 64,
      "architecture": "Maskformer",
      "dataset": "MyResidueData",
      "epochs": 25,
      }
    )

# Create the dataset
image_path = '/root/home/data/Residue_02_16/Images/'
mask_path = '/root/home/data/Residue_02_16/Masks/'

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

# Define the configurations of the transforms specific to the dataset used
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
        instance_seg = np.array(Image.open(mask_name))[..., 1]
        class_id_map = np.array(Image.open(mask_name))[..., 0]
        class_labels = np.unique(class_id_map)

        inst2class = {}
        for label in class_labels:
            instance_ids = np.unique(instance_seg[class_id_map == label])
            inst2class.update({i: label for i in instance_ids})
            
        if self.transform is not None:
            transformed = self.transform(image=image, mask=instance_seg)
            (image, instance_seg) = (transformed["image"], transformed["mask"])
            image = image.transpose(2, 0, 1)
        
        inputs = self.processor(
            [image],
            [instance_seg],
            return_tensors="pt"
        )
        inputs = {
            k:v.squeeze() if isinstance(v, torch.Tensor) else v[0] for k,v in inputs.items()
        }
        return inputs

# Distributed setup
dist.init_process_group(backend='nccl')
local_rank = int(os.getenv('LOCAL_RANK', '0'))
torch.cuda.set_device(local_rank)

# Creating image dataset
dataset = ImageSegmentationDataset(image_dir=image_path, mask_dir=mask_path, processor=processor, transform=train_val_transform)

# Split the dataset into training and validation sets
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    pixel_mask = torch.stack([example["pixel_mask"] for example in examples])
    mask_labels = [example["mask_labels"] for example in examples]
    class_labels = [example["class_labels"] for example in examples]
    return {
        "pixel_values": pixel_values,
        "pixel_mask": pixel_mask,
        "mask_labels": mask_labels,
        "class_labels": class_labels
    }

# Building the training and validation dataloader with DistributedSampler
train_sampler = DistributedSampler(train_dataset)
val_sampler = DistributedSampler(val_dataset)

train_dataloader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=False,
    sampler=train_sampler,
    collate_fn=collate_fn
)
val_dataloader = DataLoader(
    val_dataset,
    batch_size=64,
    shuffle=False,
    sampler=val_sampler,
    collate_fn=collate_fn
)

# Use GPU if available
device = torch.device(f"cuda:{local_rank}")
model.to(device)

# Wrap the model with DistributedDataParallel
model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

# Initialize Adam optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)

# Base directory to save the models
base_dir = "models"

# Set number of epochs
num_epochs = 4
for epoch in range(num_epochs):
    print(f"Epoch {epoch} | Training")
    model.train()
    train_loss, val_loss = [], []
    
    # Set sampler epoch for shuffling
    train_sampler.set_epoch(epoch)

    # Training loop
    for idx, batch in enumerate(tqdm(train_dataloader)):
        optimizer.zero_grad()
 
        outputs = model(
            pixel_values=batch["pixel_values"].to(device),
            mask_labels=[labels.to(device) for labels in batch["mask_labels"]],
            class_labels=[labels.to(device) for labels in batch["class_labels"]],
        )
        loss = outputs.loss
        train_loss.append(loss.item())
        loss.backward()
        if idx % 200 == 0:
            avg_train_loss = round(sum(train_loss)/len(train_loss), 6)
            # print(f"  Training loss: {avg_train_loss}")
            wandb.log({"training_loss": avg_train_loss, "epoch": epoch})
            
        optimizer.step()
    
    train_loss = sum(train_loss)/len(train_loss)
    model.eval()
    print(f"Epoch {epoch} | Validation")
    for idx, batch in enumerate(tqdm(val_dataloader)):
        with torch.no_grad():
            outputs = model(
                pixel_values=batch["pixel_values"].to(device),
                mask_labels=[labels.to(device) for labels in batch["mask_labels"]],
                class_labels=[labels.to(device) for labels in batch["class_labels"]],
            )
            loss = outputs.loss
            val_loss.append(loss.item())
            if idx % 200 == 0:
                avg_val_loss = round(sum(val_loss)/len(val_loss), 6)
                # print(f"  Validation loss: {avg_val_loss}")
                wandb.log({"validation_loss": avg_val_loss, "epoch": epoch})
                # print("  Validation loss: ", round(sum(val_loss)/len(val_loss), 6))
                
    
    val_loss = sum(val_loss)/len(val_loss)
    print(f"Epoch {epoch} | train_loss: {train_loss} | validation_loss: {val_loss}")

    # Log epoch end metrics
    wandb.log({
        "epoch": epoch,
        "train_loss": train_loss,
        "validation_loss": val_loss
    })
    
    # Create a new folder for this iteration
    iteration_dir = os.path.join(base_dir, f"epoch_{epoch}")
    os.makedirs(iteration_dir, exist_ok=True)
    
    # Save the model to the respective folder
    model_path = os.path.join(iteration_dir, "model")
    model.module.save_pretrained(model_path)
    processor.save_pretrained(model_path)
    wandb.log_model(model_path, "my_residue_model", aliases=[f"epoch-{epoch+1}"])

# Mark the run as finished
wandb.finish()


# # Save the trained model
# modelPath = "MaskformerModel/"
# model.module.save_pretrained(modelPath)  # Save model.module when using DDP
# processor.save_pretrained(modelPath)
