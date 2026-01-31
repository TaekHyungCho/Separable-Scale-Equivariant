import numpy as np
import torch
import torch.nn.functional as F
import math
import argparse
import os

'''
======================================================
This source code is based on 
1) "Scale-Equivariant Steerable Networks",  
paper url : https://arxiv.org/abs/1910.11093 ICLR 2019
github : https://github.com/ISosnovik/sesn

2) "DISCO: accurate Discrete Scale Convolutions", 
paper url : https://arxiv.org/abs/2106.02733 BMVC 2021
github : https://github.com/ISosnovik/disco

======================================================

We've added and modifed some codes according to its purpose.
'''

def hermite_poly(X, n):
    """Hermite polynomial of order n calculated at X
    Args:
        n: int >= 0
        X: np.array

    Output:
        Y: array of shape X.shape
    """
    coeff = [0] * n + [1]
    func = np.polynomial.hermite_e.hermeval(X, coeff)
    return func

def heremite_2DGaussian_basis(size, scales, effective_size):

    basis_tensors= []
    max_order = effective_size - 1
    max_scale = max(scales)
    for scale in scales:
        size_before_pad = int(size * scale / max_scale) // 2 * 2 + 1
        basis= one_scale_hermite_gaussian(size_before_pad,
                                            base_scale=scale,
                                            max_order=max_order)
        basis = basis[None, :, :, :]
        pad_size = (size - size_before_pad) // 2
        basis = F.pad(basis, [pad_size] * 4)[0]
        basis_tensors.append(basis)
    return torch.stack(basis_tensors, 1)

def one_scale_hermite_gaussian(size, base_scale, max_order=2):
    
    max_order = max_order
    X = np.linspace(-(size // 2), size // 2, size)
    Y = np.linspace(-(size // 2), size // 2, size)
    order_y, order_x = np.indices([max_order + 1, max_order + 1])
    scale = base_scale
    G = np.exp(-X**2 / (2 * scale**2)) / (scale)

    basis_x = [G * hermite_poly(X / scale, n) for n in order_x.ravel()]
    basis_y = [G * hermite_poly(Y / scale, n) for n in order_y.ravel()]
    basis_x = torch.Tensor(np.stack(basis_x))
    basis_y = torch.Tensor(np.stack(basis_y))
    basis = torch.bmm(basis_x[:, :, None], basis_y[:, None, :])
    return basis

def hermite_basis_varying_sigma(effective_size, scales, max_order=2, mult=2, num_funcs=None):

    num_funcs = num_funcs or effective_size ** 2
    basis_x = []
    basis_y = []

    X = torch.linspace(-(effective_size // 2), effective_size // 2, effective_size)
    Y = torch.linspace(-(effective_size // 2), effective_size // 2, effective_size)

    for scale in scales:
        G = torch.exp(-X**2 / (2 * scale**2)) / scale

        order_y, order_x = np.indices([max_order + 1, max_order + 1])
        mask = order_y + order_x <= max_order            
        bx = [G * hermite_poly(X / scale, n) for n in order_x[mask]]
        by = [G * hermite_poly(Y / scale, n) for n in order_y[mask]]

        basis_x.extend(bx)
        basis_y.extend(by)

    basis_x = torch.stack(basis_x)[:num_funcs]
    basis_y = torch.stack(basis_y)[:num_funcs]
    return torch.bmm(basis_x[:, :, None], basis_y[:, None, :])

def normalize_basis_by_min_scale(basis):
    norm = basis.pow(2).sum([2, 3], keepdim=True).sqrt()[:, [0]]
    return basis / norm

def get_basis_with_filename(dir,filename,permute=False):

    """
    Loads pre-calculated basis functions from a specified file.

    Supports both 2D Hermite-Gaussian envelope and DISCO basis functions.

    Args:
        dir (str): The directory where the basis function file is stored.
        filename (str): The name of the basis function file.
        permute (bool): Configuration for the basis function scale. 
            If True, activates scale-combined mode; otherwise, 
            utilizes the scale-isolated basis function.
    """

    fpath = os.path.join(dir,filename)
    basis = torch.load(fpath,map_location='cpu').contiguous() # [Nb, S, K, K]
    if 'disco' in filename:
        nb,ns,k,k = basis.size()
        W = hermite_basis_varying_sigma(effective_size=3,scales=[1.0,1.4,2.0])
        W = W.view(nb,-1)
        basis = (W@basis.view(nb,-1)).view(nb,ns,k,k)          
    if not permute:
        basis = normalize_basis_by_min_scale(basis)
        basis = basis.contiguous()
    else:
        basis = normalize_basis_by_min_scale(basis)
        # Reorder Nb and S for scale-combined conv filter.
        basis = basis.permute(1,0,2,3).unsqueeze(dim=0).contiguous() 
    return basis

def get_basis(size, scales, effective_size,permute=False):
    
    """
    Generates basis functions from scale-equivariant functions.

    Supports 2D Hermite-Gaussian envelopes. For DISCO basis functions, 
    pre-calculated data must be provided.

    Args:
        size (int): The kernel size of the basis functions.
        scales (list): A list of scales for the scale-equivariant basis functions.
        effective_size (int): The number of orders for the scale-equivariant basis functions (max_order + 1).
        permute (bool): Configuration for the basis function scale. 
            If True, activates scale-combined mode; otherwise, 
            utilizes the scale-isolated basis function.
    """

    basis = heremite_2DGaussian_basis(size=size,scales=scales,effective_size=effective_size) # [Nb, S, K, K]

    if not permute:
        basis = normalize_basis_by_min_scale(basis)
        basis = basis.contiguous()
    else:
        basis = normalize_basis_by_min_scale(basis)
        # Reorder Nb and S for scale-combined conv filter.
        basis = basis.permute(1,0,2,3).unsqueeze(dim=0).contiguous() 
    return basis


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Generate and save pre-calculated basis functions.')
    parser.add_argument('--size', type=int, default=5,
                         help='The kernel size of the basis functions')
    parser.add_argument('--scales', type=float, nargs='+', default=[1.0, 1.4, 2.0],
                         help='A list of scales for the scale-equivariant basis functions')
    parser.add_argument('--effective_size', type=int, default=3,
                         help='The number of orders for the scale-equivariant basis functions')
    parser.add_argument('--save_path', type=str, default='pre_calculated_basis',
                         help='Directory path where the generated basis will be stored')
    parser.add_argument('--save_name', type=str, default='basis.pt',
                         help='Filename for the saved basis tensor (should end in .pt or .pth).')
    args = parser.parse_args()

    size = args.size
    scales = args.scales
    effective_size = args.effective_size
    save_path = args.save_path
    save_name = args.save_name

    basis = heremite_2DGaussian_basis(size=size,scales=scales,effective_size=effective_size)

    if not os.path.exists(save_path):
        os.makedirs(save_path,exist_ok=True)
    
    save_full_path = os.path.join(save_path, save_name)
    torch.save(basis, save_full_path)
    print(f"Basis generated and saved to: {save_full_path}")
