# main_MAGIA.py (modified)
import argparse
import torch
from torchvision import datasets, transforms
import torch.nn.functional as F
from models.vision import LeNet, weights_init
import torch.optim as optim
from torch import nn
import random
import numpy as np
from utils import (label_to_onehot, cross_entropy_for_onehot, clip_prior, total_variation,
                   calculate_mse, calculate_rmse, calculate_psnr, calculate_ssim, calculate_reconstruction_rate, calculate_expression,
                   color_smooth_loss, frequency_loss, sigmoid_decay)
import os
import math
import itertools
import matplotlib.pyplot as plt

# ------------------------
# Global defaults (kept from original)
# ------------------------
batch_size = 64
n_cls = 100
n_iter = 300
iter_show = 10
n_plot = 10
tv = 0.005
alpha = 0.999
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------
# Selection helper
# ------------------------
def load_sel_list(path):
    """Load a newline- or comma-separated list of integers for per-iteration sel."""
    if path is None or not os.path.exists(path):
        return None
    with open(path, "r") as f:
        text = f.read().strip()
    if not text:
        return None
    # accept comma or newline separated
    parts = [p.strip() for p in text.replace("\n", ",").split(",") if p.strip() != ""]
    try:
        nums = [int(float(p)) for p in parts]
    except Exception:
        raise ValueError("sel_list file must contain integer values separated by comma/newline.")
    return nums

def clamp_sel(x):
    """Clamp selection integer into valid [1, batch_size]."""
    x = int(x)
    return max(1, min(batch_size, x))

def compute_selection(iters, args, sel_list_cache=None):
    """
    Compute selection 'sel' for current iteration according to args.sel_mode.

    Supported modes:
      - constant: fixed integer (--sel_const)
      - sigmoid: use utils.sigmoid_decay(iters, a, b)
      - linear_increase: ceil(batch_size * ((iters+1) / n_iter))
      - linear_warmup: if iters < l_iter: ceil(batch_size * ((iters+1)/l_iter)) else batch_size
      - linear_decrease: if iters < l_iter: ceil(batch_size * (1 - (iters)/l_iter)) else 1
      - fixed_list: read per-iteration sel values from file (--sel_list_path)
      - manual: alias of constant
      - fixed_value: alias of constant
      - explicit: use --sel_const
    """
    mode = args.sel_mode
    if mode == "constant" or mode == "manual" or mode == "fixed_value" or mode == "explicit":
        return clamp_sel(args.sel_const)
    if mode == "sigmoid":
        val = math.ceil(batch_size * sigmoid_decay(iters, args.a, args.b))
        return clamp_sel(val)
    if mode == "linear_increase":
        val = math.ceil(batch_size * ((iters + 1) / float(n_iter)))
        return clamp_sel(val)
    if mode == "linear_warmup":
        if iters < args.l_iter:
            val = math.ceil(batch_size * ((iters + 1) / float(args.l_iter)))
        else:
            val = batch_size
        return clamp_sel(val)
    if mode == "linear_decrease":
        if iters < args.l_iter:
            val = math.ceil(batch_size * (1 - (iters / float(args.l_iter))))
        else:
            val = 1
        return clamp_sel(val)
    if mode == "fixed_list":
        if sel_list_cache is None:
            return clamp_sel(args.sel_const)
        if iters < len(sel_list_cache):
            return clamp_sel(sel_list_cache[iters])
        else:
            # if list shorter than n_iter, use last value
            return clamp_sel(sel_list_cache[-1])
    # fallback
    return clamp_sel(args.sel_const)

# ------------------------
# Argument parsing
# ------------------------
parser = argparse.ArgumentParser(description='Implementation with selectable selection schemes.')
parser.add_argument('--indices', type=str, default=None,
                    help='Comma-separated indices for leaking images on CIFAR-100. If omitted, random indices will be generated.')
parser.add_argument('--sel_mode', type=str, default='linear_warmup',
                    choices=['constant', 'sigmoid', 'linear_increase', 'linear_warmup', 'linear_decrease', 'fixed_list', 'manual', 'fixed_value', 'explicit'],
                    help='Selection mode for sampling a sub-batch during optimization.')
parser.add_argument('--sel_const', type=int, default=2,
                    help='Constant selection value used by constant/manual/fixed_value modes.')
parser.add_argument('--sel_list_path', type=str, default=None,
                    help='Path to a file containing per-iteration sel values (comma or newline separated). Used with fixed_list mode.')
parser.add_argument('--l_iter', type=int, default=150,
                    help='Warmup/decay length used by linear_warmup and linear_decrease modes.')
parser.add_argument('--a', type=float, default=-0.1,
                    help='Parameter a for sigmoid decay when using sigmoid mode.')
parser.add_argument('--b', type=float, default=200.0,
                    help='Parameter b for sigmoid decay when using sigmoid mode.')
parser.add_argument('--n_iter', type=int, default=n_iter,
                    help='Number of optimization iterations (overrides the default n_iter).')
args = parser.parse_args()

# allow overriding global n_iter by arg
n_iter = args.n_iter

# ------------------------
# Data / indices setup
# ------------------------
print(f"Running on {device}")

dst = datasets.CIFAR100(root="~/.torch", download=True, transform=transforms.ToTensor())

# prepare indices: if not provided, sample randomly
if args.indices is None:
    random_numbers = [random.randint(0, len(dst) - 1) for _ in range(batch_size)]
    indices = random_numbers
else:
    indices = [int(idx.strip()) % len(dst) for idx in args.indices.split(",")]

gt_data = torch.stack([dst[i][0] for i in indices]).to(device)
gt_label = torch.tensor([dst[i][1] for i in indices]).to(device)
gt_onehot_label = label_to_onehot(gt_label, n_cls)

# save random index file if indices were random
if args.indices is None:
    result_string = ','.join(str(num) for num in indices)
    with open('random_numbers_sgd.txt', 'w') as file:
        file.write(result_string)

# ------------------------
# Visualization of ground truth (kept)
# ------------------------
tp = transforms.ToTensor()
tt = transforms.ToPILImage()

num_images = len(indices)
num_columns = 5
num_rows = (num_images + num_columns - 1) // num_columns
figsize = (12, 4 * num_rows)
plt.figure(figsize=figsize)

for i in range(num_images):
    plt.subplot(num_rows, num_columns, i + 1)
    plt.imshow(tt(gt_data[i].cpu()))
    plt.title(f"Image {indices[i]}")
    plt.axis('off')

plt.tight_layout()
output_dir = "recovery_history_sgd"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "original.png")
plt.savefig(output_path, dpi=300)
plt.close()

# ------------------------
# Model & gradient setup
# ------------------------
net = LeNet().to(device)
torch.manual_seed(111)
net.apply(weights_init)
criterion = cross_entropy_for_onehot

pred = net(gt_data)
y = criterion(pred, gt_onehot_label)
dy_dx = torch.autograd.grad(y, net.parameters(), create_graph=False)
original_dy_dx = [_.detach().clone() for _ in dy_dx]

# ------------------------
# Dummy inputs and optimizer
# ------------------------
dummy_data = torch.randn_like(gt_data, device=device)
dummy_data = torch.clamp(dummy_data, min=0, max=255)
dummy_data.requires_grad = True

dummy_label = torch.randn(gt_onehot_label.size()).to(device).requires_grad_(True)
optimizer = optim.LBFGS([dummy_data, dummy_label])

# history container
history = [[None for _ in range(n_iter // 10)] for _ in range(dummy_data.size(0))]

# load sel list if requested
sel_list_cache = None
if args.sel_mode == "fixed_list" and args.sel_list_path is not None:
    sel_list_cache = load_sel_list(args.sel_list_path)

# ------------------------
# Optimization loop
# ------------------------
for iters in range(n_iter):
    sel = compute_selection(iters, args, sel_list_cache=sel_list_cache)

    def closure():
        optimizer.zero_grad()
        dummy_pred = net(dummy_data)
        dummy_onehot_label = F.softmax(dummy_label, dim=-1)
        cr_loss = criterion(dummy_pred, dummy_onehot_label)
        dummy_loss = cr_loss
        dummy_dy_dx = torch.autograd.grad(dummy_loss, net.parameters(), create_graph=True)
        grad_diff = sum(((gx - gy) ** 2).sum() for gx, gy in zip(dummy_dy_dx, original_dy_dx))

        rand_idx = torch.randperm(batch_size)[:sel]
        dummy_data_comb = dummy_data[rand_idx]
        dummy_label_comb = dummy_label[rand_idx]
        dummy_label_comb_sm = F.softmax(dummy_label_comb, dim=-1)
        dummy_pred_comb = net(dummy_data_comb)
        comb_loss = criterion(dummy_pred_comb, dummy_label_comb_sm)
        dummy_dy_dx_comb = torch.autograd.grad(comb_loss, net.parameters(), create_graph=True)

        # note: original code used (gx/sel - gy/batch_size) ** 2 .mean()
        grad_diff_comb = sum(((gx / sel - gy / batch_size) ** 2).mean() for gx, gy in zip(dummy_dy_dx_comb, original_dy_dx))

        total_loss = (alpha * calculate_expression(batch_size, sel) * grad_diff +
                      (1 - alpha) * calculate_expression(batch_size, sel) * grad_diff_comb +
                      tv * total_variation(dummy_data))
        total_loss.backward()
        return total_loss

    optimizer.step(closure)

    if iters % n_plot == 0:
        current_loss = closure()
        print(f"Iteration {iters}, Loss: {current_loss.item():.9f}, sel={sel}, sel_mode={args.sel_mode}")

        dummy_images = dummy_data.clone().detach().cpu()
        for idx in range(batch_size):
            history[idx][iters // n_plot] = dummy_images[idx]

# ------------------------
# Evaluation metrics and plotting
# ------------------------
dummy_data = dummy_data.clone().detach().cpu()
original_data = gt_data.clone().detach().cpu()

batch_metrics = []
for i in range(dummy_data.size(0)):
    d_img = dummy_data[i]
    best_metrics = None
    best_index = None
    max_similarity = float('-inf')
    for j in range(original_data.size(0)):
        o_img = original_data[j]
        mse = calculate_mse(d_img, o_img)
        rmse = calculate_rmse(d_img, o_img)
        psnr = calculate_psnr(d_img, o_img)
        ssim_value = calculate_ssim(d_img, o_img)
        recon_rate = calculate_reconstruction_rate(d_img, o_img)
        if ssim_value > max_similarity:
            max_similarity = ssim_value
            best_metrics = {
                'Idx': j,
                'MSE': mse,
                'RMSE': rmse,
                'PSNR': psnr,
                'SSIM': ssim_value,
                'Reconstruction Rate': recon_rate
            }
            best_index = j
    batch_metrics.append(best_metrics)

for metrics in batch_metrics:
    print(f"Generated Image Best Match with Original Image {metrics['Idx']}:")
    for metric_name, value in metrics.items():
        if metric_name != 'Idx':
            print(f"  {metric_name}: {value:.6f}")

metrics_keys = ['MSE', 'RMSE', 'PSNR', 'SSIM', 'Reconstruction Rate']
metrics_values = {key: [metrics[key] for metrics in batch_metrics] for key in metrics_keys}
mean_metrics = {key: np.mean(values) for key, values in metrics_values.items()}
std_metrics = {key: np.std(values) for key, values in metrics_values.items()}

print("\nOverall Metrics:")
for key in metrics_keys:
    print(f"  {key}: Mean = {mean_metrics[key]:.6f}, Std = {std_metrics[key]:.6f}")

plt.figure(figsize=(n_iter // n_plot * 2, batch_size * 2))
for idx in range(batch_size):
    for iter_idx in range(n_iter // n_plot):
        plt.subplot(batch_size, n_iter // n_plot, idx * (n_iter // n_plot) + iter_idx + 1)
        plt.imshow(tt(history[idx][iter_idx]))
        plt.axis('off')
plt.tight_layout()

output_dir = "recovery_history_sgd"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "recovery_history.png")
plt.savefig(output_path, dpi=300)
plt.close()

print(f"Recovery history saved to {output_path}")
