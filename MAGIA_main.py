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
                   calculate_mse, calculate_rmse, calculate_psnr, calculate_ssim,calculate_reconstruction_rate, calculate_expression,
                   color_smooth_loss, frequency_loss, sigmoid_decay)
import os
import math
import itertools
\1
def compute_sel(iters, batch_size, n_iter, mode, l_iter, k, a, b):
    """
    Compute subset size S (sel) according to selection strategy.
    Ensures 1 <= sel <= batch_size.
    Strategies mirror those discussed in the paper's appendix.
    """
    if mode == 'const1':
        sel = 1
    elif mode == 'constk':
        sel = int(k)
    elif mode == 'linear_inc_l':
        if iters < l_iter:
            sel = int((batch_size * ((iters + 1) / max(1, l_iter))) + 0.9999)
        else:
            sel = batch_size
    elif mode == 'linear_inc_total':
        sel = int((batch_size * ((iters + 1) / max(1, n_iter))) + 0.9999)
    elif mode == 'linear_dec_l':
        if iters < l_iter:
            sel = int((batch_size * (1.0 - (iters / max(1, l_iter)))) + 0.9999)
        else:
            sel = 1
    elif mode == 'linear_dec_total':
        sel = int((batch_size * (1.0 - (iters / max(1, n_iter)))) + 0.9999)
    elif mode == 'sigmoid':
        # sigmoid_decay is imported from utils
        sel = int((batch_size * sigmoid_decay(iters, a, b)) + 0.9999)
    else:
        # Fallback to constk
        sel = int(k)

    sel = max(1, min(batch_size, sel))
    return sel


batch_size = 64
n_cls = 100
n_iter = 300
iter_show = 10
n_plot = 10
tv = 0.005 # CIFAR-100 4 0.005
clip = 1
alpha = 0.999
cos_v = 0.01
xi = 0.01
cos = 1 / batch_size
beta = 0.01
comb = 1 / batch_size

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on {device}")


dst = datasets.CIFAR100(root="~/.torch", download=True, transform=transforms.ToTensor())
tp = transforms.ToTensor()
tt = transforms.ToPILImage()

random_numbers = [random.randint(0, len(dst) - 1) for _ in range(batch_size)]
# random_numbers = [549997,332569,663860,398317,296769,527429,132550,611531,155040,125532,561360,576710,117533,473600,526087,10752,408637,434702,199269,182549,167124,301615,141396,460286,193776,342199,676998,420329,63713,435601,234911,60931,618828,690216,2005,193647,674370,687001,523755,67043]
### Random number of the Figure 2 (a) in the text
result_string = ','.join(str(num) for num in random_numbers)

with open('random_numbers_sgd.txt', 'w') as file:
    file.write(result_string)


parser = argparse.ArgumentParser(description='Deep Leakage from Gradients with Batch Support.')
parser.add_argument('--sel_mode', type=str, default='constk',
                    choices=['const1','constk','linear_inc_l','linear_inc_total','linear_dec_l','linear_dec_total','sigmoid'],
                    help='Selection strategy for subset size S.')
parser.add_argument('--sel_k', type=int, default=2, help='Constant S when sel_mode=constk.')
parser.add_argument('--sel_l_iter', type=int, default=100, help='Warmup/decay length for linear_inc_l or linear_dec_l.')
parser.add_argument('--sig_a', type=float, default=-0.1, help='Sigmoid decay parameter a (used when sel_mode=sigmoid).')
parser.add_argument('--sig_b', type=float, default=200.0, help='Sigmoid decay parameter b (used when sel_mode=sigmoid).')



indices = [int(idx.strip()) % len(dst) for idx in args.indices.split(",")]
gt_data = torch.stack([dst[i][0] for i in indices]).to(device)
gt_label = torch.tensor([dst[i][1] for i in indices]).to(device)
gt_onehot_label = label_to_onehot(gt_label, n_cls)




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
plt.show()
plt.close()


net = LeNet().to(device)
torch.manual_seed(111)
net.apply(weights_init)
criterion = cross_entropy_for_onehot



pred = net(gt_data)


y = criterion(pred, gt_onehot_label)
dy_dx = torch.autograd.grad(y, net.parameters(), create_graph=False)
original_dy_dx = [_.detach().clone() for _ in dy_dx]






dummy_data = torch.randn_like(gt_data, device=device)
dummy_data = torch.clamp(dummy_data, min=0, max=255)
dummy_data.requires_grad = True  # Set if needed for gradients


dummy_label = torch.randn(gt_onehot_label.size()).to(device).requires_grad_(True)




optimizer = optim.LBFGS([dummy_data, dummy_label])


# history = []


history = [[None for _ in range(n_iter // 10)] for _ in range(dummy_data.size(0))]

\1    # Compute subset size 'sel' according to the chosen selection strategy
    sel = compute_sel(
        iters=iters,
        batch_size=batch_size,
        n_iter=n_iter,
        mode=args.sel_mode,
        l_iter=args.sel_l_iter,
        k=args.sel_k,
        a=args.sig_a,
        b=args.sig_b,
    )
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
        grad_diff_comb = sum(((gx / sel - gy / batch_size) ** 2).mean() for gx, gy in zip(dummy_dy_dx_comb, original_dy_dx))




        total_loss = (alpha * calculate_expression(batch_size, sel) * grad_diff + (1 - alpha) * calculate_expression(batch_size, sel) * grad_diff_comb +
                      tv * total_variation(dummy_data))


        total_loss.backward()
        return total_loss

    optimizer.step(closure)


    if iters % n_plot == 0:
        current_loss = closure()
        print(f"Iteration {iters}, Loss: {current_loss.item():.9f}")
        dummy_images = dummy_data.clone().detach().cpu()
        for idx in range(batch_size):
            history[idx][iters // n_plot] = dummy_images[idx]

############################################################

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
        # plt.title(f"Idx={indices[idx]}, Iter={iter_idx * n_plot}")
        plt.axis('off')
plt.tight_layout()


output_dir = "recovery_history_sgd"
os.makedirs(output_dir, exist_ok=True)


output_path = os.path.join(output_dir, "recovery_history.png")
plt.savefig(output_path, dpi=300)
# plt.show()
plt.close()

print('')
