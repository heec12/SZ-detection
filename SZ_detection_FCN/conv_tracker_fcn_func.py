import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import numpy as np
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

# Reusable block: Conv3x3 -> BN -> ReLU -> SE
class ConvBNReLUSE(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.bn   = nn.BatchNorm2d(out_ch)
        self.se   = SEBlock(out_ch)
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x, inplace=True)
        x = self.se(x)
        return x

# ---------------- FCN8: add 4 extra 3x3 convs -----------------
class FCN(nn.Module):
    """Total convs = 8 (adds: two extra 64ch at /4, two extra 32ch at /1)."""
    def __init__(self, dropout_p=0.5, align_corners=False):
        super().__init__()
        self.conv1 = ConvBNReLUSE(3, 32)          # /1
        self.conv2 = ConvBNReLUSE(32, 64)         # /2
        self.mid64a = ConvBNReLUSE(64, 64)        # EXTRA @ /4
        self.mid64b = ConvBNReLUSE(64, 64)        # EXTRA @ /4
        self.dec32a = ConvBNReLUSE(64, 32)        # /2 -> /1
        self.dec32b = ConvBNReLUSE(32, 32)        # EXTRA @ /1
        self.dec32c = ConvBNReLUSE(32, 32)        # EXTRA @ /1
        self.conv_out = nn.Conv2d(32, 1, 1)

        self.pool     = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=align_corners)
        self.dropout  = nn.Dropout(dropout_p)

    def forward(self, x):
        x = self.conv1(x)          # /1
        x = self.pool(x)           # /2
        x = self.conv2(x)          # /2
        x = self.pool(x)           # /4

        x = self.mid64a(x)         # EXTRA /4
        x = self.mid64b(x)         # EXTRA /4

        x = self.upsample(x)       # /2
        x = self.dec32a(x)         # 64->32
        x = self.upsample(x)       # /1
        x = self.dec32b(x)         # EXTRA /1
        x = self.dec32c(x)         # EXTRA /1
        x = self.dropout(x)
        x = self.conv_out(x)
        return x

# Implementing "early stopping"
class EarlyStopping:
    def __init__(self, patience=8, min_delta=1e-3, mode="min", ckpt_path="best.pt"):
        assert mode in ("min", "max")
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.ckpt_path = ckpt_path
        self.best = None
        self.bad_epochs = 0

    def _is_better(self, current, best):
        if self.mode == "min":
            return current < (best - self.min_delta)
        else:
            return current > (best + self.min_delta)

    def step(self, current_value, model):
        if self.best is None or self._is_better(current_value, self.best):
            self.best = current_value
            self.bad_epochs = 0
            torch.save(model.state_dict(), self.ckpt_path)
            return False  # don't stop
        self.bad_epochs += 1
        return self.bad_epochs >= self.patience  # True => stop


# Function for manual data augmentation
def augment_data(x, y):
    # x: [C,H,W], y: [H,W]
    if random.random() > 0.5:  # horizontal (width axis)
        x = torch.flip(x, [2])
        y = torch.flip(y, [1])
    if random.random() > 0.5:  # vertical (height axis)
        x = torch.flip(x, [1])
        y = torch.flip(y, [0])
    if random.random() > 0.5:
        ang = random.randint(-10, 10)
        x = rotate_tensor_img(x, ang)      # bilinear
        y = rotate_tensor_mask(y, ang)     # nearest
    return x, y

def rotate_tensor_img(x, angle_deg):
    angle = np.deg2rad(angle_deg)
    theta = torch.tensor([[np.cos(angle), -np.sin(angle), 0],
                          [np.sin(angle),  np.cos(angle), 0]], dtype=torch.float, device=x.device).unsqueeze(0)
    N,C,H,W = 1, x.shape[0], x.shape[1], x.shape[2]
    grid = nn.functional.affine_grid(theta, size=(N,C,H,W), align_corners=False)
    return nn.functional.grid_sample(x.unsqueeze(0), grid, mode='bilinear',
                                     padding_mode='zeros', align_corners=False).squeeze(0)

def rotate_tensor_mask(y, angle_deg):
    angle = np.deg2rad(angle_deg)
    theta = torch.tensor([[np.cos(angle), -np.sin(angle), 0],
                          [np.sin(angle),  np.cos(angle), 0]], dtype=torch.float, device=y.device).unsqueeze(0)
    N,C,H,W = 1, 1, y.shape[0], y.shape[1]
    grid = nn.functional.affine_grid(theta, size=(N,C,H,W), align_corners=False)
    return nn.functional.grid_sample(y.unsqueeze(0).unsqueeze(0), grid, mode='nearest',
                                     padding_mode='zeros', align_corners=False).squeeze(0).squeeze(0)

# Combined BCEWithLogitsLoss and Dice Loss
def combined_loss(pred, target, smooth=1.0):
    # pred: [B,1,H,W]
    # target: [B,1,H,W] or [B,H,W]
    if target.dim() == 3:
        target = target.unsqueeze(1)          # -> [B,1,H,W]
    elif target.dim() == 2:
        target = target.unsqueeze(0).unsqueeze(0)  # handle single image path

    target = target.to(pred.dtype)

    # BCE
    bce = F.binary_cross_entropy_with_logits(pred, target)

    # Dice
    probs = torch.sigmoid(pred)
    intersection = (probs * target).sum(dim=(1,2,3))
    denom = probs.sum(dim=(1,2,3)) + target.sum(dim=(1,2,3))
    dice = 1 - ((2.0 * intersection + smooth) / (denom + smooth)).mean()

    return bce + dice

# Function to set the default output probability equals the empirical positive rate
# compute the empirical positive rate
def compute_positive_prior(Y_train):
    """
    Y_train: torch.Tensor of masks, shape [N, 1, H, W] or [N, H, W], values in {0,1}
    returns float p in (0,1)
    """
    with torch.no_grad():
        y = Y_train
        if y.dim() == 3:  # [N,H,W] -> [N,1,H,W]
            y = y.unsqueeze(1)
        p = y.float().mean().item()
        # clamp to avoid inf logits
        p = max(min(p, 1 - 1e-6), 1e-6)
    return p

# set the default probability
def bias_init_final_logits_to_prior(model, p):
    """
    Find the last Conv2d with out_channels==1 and set:
      bias = log(p/(1-p))
      weight = 0
    """
    prior_logit = math.log(p / (1.0 - p))
    last = None
    for m in model.modules():
        if isinstance(m, nn.Conv2d) and m.out_channels == 1:
            last = m
    if last is None:
        raise RuntimeError("No final Conv2d(out_channels=1) found.")
    with torch.no_grad():
        if last.bias is None:
            last.bias = nn.Parameter(torch.empty(1, device=next(model.parameters()).device))
        last.bias.fill_(prior_logit)
        last.weight.zero_()  # optional but helps start “flat”

# Function to train the model with early stopping
def train_model_with_early_stopping(
    model, criterion, optimizer, scheduler,
    X_train, Y_train, X_val, Y_val, device,
    epochs=100, patience=8, min_delta=1e-3, ckpt_path="./best_fcn.pt"
):
    early = EarlyStopping(patience=patience, min_delta=min_delta, mode="min", ckpt_path=ckpt_path)

    model.train()
    Ntrain = X_train.size(0)

    for epoch in range(1, epochs+1):
        running_loss = 0.0

        # ---- training (sample-by-sample) ----
        for i in range(Ntrain):
            # x: [1,3,H,W]; y: [1,1,H,W] Squeeze batch dim to use the augment function
            x = X_train[i]      # [3,H,W]
            y = Y_train[i,0]    # [H,W]  Take channel 0 so the augment_data works as written

            # apply augmentation (function expects [C,H,W] and [H,W])
            xa, ya = augment_data(x, y)

            # restore shapes for model: [1,3,H,W] and [1,1,H,W]
            xb = xa.unsqueeze(0).to(device).float()
            yb = ya.unsqueeze(0).unsqueeze(0).to(device).float()

            optimizer.zero_grad()
            outputs = model(norm(xb))                 # [1,1,H,W] expected
            loss = criterion(outputs, yb)       # combined loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item()

        # step the scheduler
        scheduler.step()

        # ---- validation pass (no aug) ----
        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            Nval = X_val.size(0)
            for j in range(Nval):
                xv = X_val[j].unsqueeze(0).to(device).float()     # [1,3,H,W]
                yv = Y_val[j].to(device).float()                  # [1,1,H,W]
                outv = model(norm(xv))
                vloss = criterion(outv, yv)
                val_loss += vloss.item()
            val_loss /= max(1, Nval)
        model.train()

        print(f"Epoch {epoch:03d} | train_loss={running_loss / Ntrain:.4f}  val_loss={val_loss:.4f}")

        # ---- early stopping on validation loss (minimization) ----
        if early.step(val_loss, model):
            print(f"Early stopping at epoch {epoch}. Best val_loss={early.best:.4f}")
            break

    # restore best weights
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    return model

# New function to evaluate the model
@torch.no_grad()
def dice_from_logits_binary(logits, target, eps=1e-6):
    # logits: [1,1,H,W], target: [1,1,H,W]
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).float()
    inter = (preds * target).sum()
    union = preds.sum() + target.sum()
    return ((2*inter + eps) / (union + eps)).item()

@torch.no_grad()
def evaluate_one(model, criterion, x, y, device):
    """
    x: [1,3,H,W] (batched 1)
    y: [H,W] or [1,1,H,W]
    Returns: loss(float), logits [1,1,H,W], dice(float)
    """
    model.eval()
    xb = x.to(device).float()
    if y.ndim == 2:            # [H,W] -> [1,1,H,W]
        yb = y.unsqueeze(0).unsqueeze(0).to(device).float()
    elif y.ndim == 3:          # [1,H,W] -> [1,1,H,W]
        yb = y.unsqueeze(0).to(device).float()
    else:                      # already [1,1,H,W]
        yb = y.to(device).float()

    logits = logits = model(norm(xb))         # [1,1,H,W]
    loss = criterion(logits, yb).item()
    dice = dice_from_logits_binary(logits, yb)
    return loss, logits, dice


# Function to evaluate the model
def evaluate_model(model, criterion, X_eval, y_eval, device):
    model.eval()  # Set the model to evaluation mode
    total_loss = 0.0
    with torch.no_grad():
        for x, y in zip(X_eval, y_eval):
            x = x.to(device).float()
            y = y.to(device).float()

            outputs = model(x.unsqueeze(0))
            outputs = outputs.squeeze() # To match the dimension
            loss = criterion(outputs, y)
            total_loss += loss.item()

    average_loss = total_loss / len(X_eval)
    return average_loss, outputs

# Post-processing function
def post_process(outputs, threshold=0.5):
    outputs = torch.sigmoid(outputs)
    outputs = (outputs > threshold).float()
    return outputs

# Function to create a predicted SZ
def plot_results(y_matrix, title):
    plt.figure(figsize=(8, 2))
    plt.imshow(y_matrix, vmin=0, cmap='binary')
    plt.title(title)
    #plt.colorbar(label='Boolean Value')
    plt.tight_layout()
    plt.show()

# Function to create the original RGB image
def plot_rgb(loaded_tensor_rgb, title):
    np_array = loaded_tensor_rgb.cpu().numpy()
    np_array = np_array / 255.0  # normalize to 0–1
    plt.figure(figsize=(8, 2))
    plt.title(title)
    plt.imshow(np_array)
    plt.tight_layout()
    plt.show()

# Function to create a predicted SZ
def plot_actual(y_matrix, title):
    y_matrix = y_matrix.squeeze(0)
    plt.figure(figsize=(8, 2))
    plt.imshow(y_matrix, vmin=0, cmap='binary')
    plt.title(title)
    #plt.colorbar(label='Boolean Value')
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

def find_edges_and_midpoints(top_row_tensor, ha_temp=None, width=512):
    """
    Detect rising/falling edges and their midpoints along a circular top row.

    Parameters
    ----------
    top_row_tensor : torch.Tensor
        1-D tensor of shape [W] with values {0,1}.
    ha_temp : any
        Optional label (e.g., timestep) for logging.
    width : int
        Total horizontal width (default = 512).

    Returns
    -------
    top_row : torch.Tensor
        Original binary top row.
    rising_edges : list[int]
        Indices where signal goes 0→1. Left boundary of SZ
    falling_edges : list[int]
        Indices where signal goes 1→0. Right boundary of SZ
    mid_points : list[int]
        Midpoints between each rise/fall pair, wrapped into [0, width-1].
    """
    tr_np = top_row_tensor.detach().cpu().numpy().astype(np.int8)
    d = np.diff(tr_np)
    rising_edges  = np.where(d == 1)[0] + 1
    falling_edges = np.where(d == -1)[0] + 1

    if len(rising_edges) == 0 or len(falling_edges) == 0:
        print(f"No edges detected for {ha_temp}")
        return top_row_tensor, [], [], []

    rising_edges  = rising_edges.tolist()
    falling_edges = falling_edges.tolist()

    f_for_pairing = falling_edges.copy()
    # (1) Shift only the first falling edge if it precedes the first rising edge
    if rising_edges and f_for_pairing and rising_edges[0] > f_for_pairing[0]:
        f_for_pairing[0] += width
        # (2) Reorient the list so this shifted element moves to the end
        f_for_pairing = f_for_pairing[1:] + f_for_pairing[:1]

    # --- compute midpoints ---
    mid_points = []
    i, j = 0, 0
    while i < len(rising_edges) and j < len(f_for_pairing):
        if f_for_pairing[j] <= rising_edges[i]:
            j += 1
            continue
        mp = (rising_edges[i] + f_for_pairing[j]) // 2
        if mp >= width:
            mp -= width  # wrap back into [0,width-1]
        mid_points.append(int(mp))
        i += 1
        j += 1

    return top_row_tensor, rising_edges, falling_edges, mid_points

def _safe_torch_load(path):
    try:
        return torch.load(path, weights_only=True)
    except TypeError:
        return torch.load(path)

def apply_fcn(ha_temp, model_path, device: torch.device):
    input_temp_ha = 'rgb_ha_temp' + "{:0>4d}".format(int(ha_temp))
    filename_rgb = f'{model_path}{input_temp_ha}.pt'

    # --- load input, shape [H,W,3] -> [1,3,H,W] ---
    X_hw3 = _safe_torch_load(filename_rgb).float()
    if X_hw3.ndim != 3 or X_hw3.shape[-1] != 3:
        raise ValueError(f"Expected [H,W,3] RGB tensor, got {tuple(X_hw3.shape)}")

    X = X_hw3.permute(2, 0, 1).unsqueeze(0).contiguous().to(device)  # [1,3,H,W]

    # --- per-image normalization ---
    with torch.no_grad():
        mean = X.mean(dim=(0,2,3), keepdim=True)
        std  = X.std (dim=(0,2,3), keepdim=True).clamp_min(1e-6)
        Xn   = (X - mean) / std

    # --- build model & load weights ---
    model = FCN().to(device)
    weights_file = './trained_model_fcn_best.pt'  # your saved model weights
    state = torch.load(weights_file, map_location=device)
    model.load_state_dict(state)
    model.eval()

    # --- forward pass ---
    with torch.no_grad():
        logits = model(Xn)                    # [1,1,H,W] (assumed logits)
        probs  = torch.sigmoid(logits)        # [1,1,H,W]
        outputs = (probs >= 0.5).float()      # [1,1,H,W] in {0,1}

    # --- top row & edge detection ---
    # top_row: shape [W] with values 0/1
    top_row = outputs.squeeze(0).squeeze(0)[0, :]  # [W]

    top_row, rising_edges, falling_edges, mid_points = find_edges_and_midpoints(top_row, ha_temp)

    return top_row, rising_edges, falling_edges, mid_points
