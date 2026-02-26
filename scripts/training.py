from monai.networks.nets import UNet
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from func.dataset import TrainDataset
from func.functions import *
import datetime
import random
from torch.utils.tensorboard import SummaryWriter 

print("script starting...")

#SETTINGS
patch_size = (32, 32, 32)
stride = (16, 16, 16)
target_shape = (192, 224, 192)

spatial_dims=3
in_channels=2
out_channels=1
net_channels = (32, 64, 128, 256, 512, 1024)
net_strides = (2, 2, 2, 2, 2)
net_res_units = 10
norm=None

loss_fn = nn.MSELoss()
batch_size = 32
num_epochs = 100
note = "Session 1"
timestamp = datetime.datetime.now().isoformat()

print(note)
print("Start at:", timestamp)

DATA_DIR = ... #path to folder with data
REPO_ROOT = ... #path to this repo                                                   
print(f"Data directory: {DATA_DIR}")

(REPO_ROOT / "results" / "tensorboard_logs").mkdir(parents=True, exist_ok=True)
writer = SummaryWriter(log_dir= REPO_ROOT / "results" / "tensorboard_logs" / timestamp)

#load file paths to lists - ENSURE CORRECT LOADING OF DATA!
t1_files = sorted(DATA_DIR.rglob("*T1w.nii.gz")) # isotropic t1w images
t2_files = sorted(DATA_DIR.rglob("*T2w.nii.gz")) # isotropic t2w images (ground truth)
t2_lr_files = sorted(DATA_DIR.rglob("*T2w_LR.nii.gz")) #anisotropic t2w images

print(f"T1 files: {len(t1_files)}, T2 files: {len(t2_files)}, T2 LR files: {len(t2_lr_files)}")

files = list(zip(t1_files, t2_files, t2_lr_files)) # list of triplets of file paths
train, val, test = split_dataset(files) # split into train, val and test sets, default split is 70% train, 15% val, 15% test

#SHUFFLE DATA
random.shuffle(train)
random.shuffle(val)
random.shuffle(test)

print(f"Train size: {len(train)}, Val size: {len(val)}, Test size: {len(test)}")

# GPU/CPU detection
import os
slurm_gpus = int(os.environ.get('SLURM_GPUS_ON_NODE', '0'))
has_gpu = torch.cuda.is_available() and slurm_gpus > 0 and torch.cuda.device_count() > 0
device = torch.device("cuda" if has_gpu else "cpu")
print(f"Using: {device} (SLURM GPUs: {slurm_gpus})")

print("Starting training...")
#NETWORK TRAINING
train_dataset = TrainDataset(train, patch_size, stride, target_shape)
train_loader = DataLoader(train_dataset, batch_size, shuffle=False)
val_loader = DataLoader(TrainDataset(val, patch_size, stride, target_shape), batch_size, shuffle=False)

print(f"Number of training batches: {len(train_loader)}")
net = UNet(
    spatial_dims=spatial_dims,
    in_channels=in_channels,
    out_channels=out_channels,
    channels=net_channels,
    strides=net_strides,
    num_res_units=net_res_units,
    norm=norm,
)
net.to(device, dtype=torch.float32)
loss_list = []
val_loss_list = []
optimizer = optim.Adam(net.parameters(), lr=1e-4)
print("Network initialized")

best_val_loss = float('inf')

for epoch in range(num_epochs):
    epoch_start_time = datetime.datetime.now()
    #TRAINING
    net.train()
    train_loss = 0.0
    for batch in train_loader:
        input1, input2, target = batch
        inputs = torch.stack([input1, input2], dim=1).to(device, dtype=torch.float32, non_blocking=True)
        target = target.unsqueeze(1).to(device, dtype=torch.float32, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        outputs = net(inputs)
        loss = loss_fn(outputs, target)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item() * inputs.size(0)

    #VALIDATION
    net.eval()
    with torch.no_grad():
        val_loss = 0.0
        for batch in val_loader:
            input1, input2, target = batch
            inputs = torch.stack([input1, input2], dim=1).to(device, dtype=torch.float32, non_blocking=True)
            target = target.unsqueeze(1).to(device, dtype=torch.float32, non_blocking=True) 

            outputs = net(inputs)
            loss = loss_fn(outputs, target)
            val_loss += loss.item() * inputs.size(0)

    epoch_train_loss = train_loss / len(train_loader.dataset)
    loss_list.append(epoch_train_loss)
    epoch_val_loss = val_loss / len(val_loader.dataset)
    val_loss_list.append(epoch_val_loss)

    writer.add_scalar('Loss/Train', epoch_train_loss, epoch)
    writer.add_scalar('Loss/Validation', epoch_val_loss, epoch)

    #save the best model based on validation loss
    if epoch_val_loss < best_val_loss:
        best_val_loss = epoch_val_loss
        (REPO_ROOT / "results" / "weights").mkdir(parents=True, exist_ok=True)
        torch.save(net.state_dict(), REPO_ROOT / "results" / "weights" / f"{timestamp}_model_weights.pth")
        best_epoch = epoch + 1 # Store the best epoch number
        writer.add_text('Best Model', f'New best model saved at epoch {best_epoch} with val loss {best_val_loss:.4f}', best_epoch)

    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}")
    print(f"Epoch duration: {(datetime.datetime.now() - epoch_start_time).total_seconds():.2f} seconds")

# SAVE RESULTS

row_dict = {
    "note": note,
    "weights": f"{timestamp}_model_weights.pth",
    "start time": timestamp,
    "end time": datetime.datetime.now().isoformat(),
    "train_size": len(train),
    "val_size": len(val),
    "test_size": len(test),
    "patch_size": patch_size,
    "stride": stride,
    "target_shape": target_shape,
    "normalization": "min-max, percentile clipping",
    "model": "MONAI 3D U-Net",
    "net spatial_dims": spatial_dims,
    "net in_channels": in_channels,
    "net out_channels": out_channels,
    "net channels": net_channels,
    "net strides": net_strides,
    "net num_res_units": net_res_units,
    "loss function": "MSE Loss",
    "net norm": norm,
    "max num of epochs": num_epochs,
    "best_epoch": best_epoch,
    "batch_size": batch_size,
    "optimizer": "Adam",
    "learning_rate": optimizer.param_groups[0]['lr'],
    "loss_list": loss_list,
    "val_loss_list": val_loss_list,
}

#create outputs directory if it doesn't exist
append_row(REPO_ROOT / "results" / "training_info.csv", row_dict)
print("End at:", datetime.datetime.now().isoformat())
writer.close()