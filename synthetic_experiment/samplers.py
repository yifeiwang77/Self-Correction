import math

import torch


class DataSampler:
    def __init__(self, n_dims):
        self.n_dims = n_dims

    def sample_xs(self):
        raise NotImplementedError


def get_data_sampler(data_name, n_dims, **kwargs):
    names_to_classes = {
        "gaussian": GaussianSampler,
        "fixed_gaussian":FixedGaussianSampler,
        "mixed_gaussian":MixedGaussianSampler,
    }
    if data_name in names_to_classes:
        sampler_cls = names_to_classes[data_name]
        return sampler_cls(n_dims, **kwargs)
    else:
        print("Unknown sampler")
        raise NotImplementedError


def sample_transformation(eigenvalues, normalize=False):
    n_dims = len(eigenvalues)
    U, _, _ = torch.linalg.svd(torch.randn(n_dims, n_dims))
    t = U @ torch.diag(eigenvalues) @ torch.transpose(U, 0, 1)
    if normalize:
        norm_subspace = torch.sum(eigenvalues**2)
        t *= math.sqrt(n_dims / norm_subspace)
    return t

class MixedGaussianSampler(DataSampler):
    def __init__(self, n_dims, bias=None, scale=None):
        super().__init__(n_dims)
        self.bias = bias
        self.scale = scale

    def sample_xs(self, n_points, b_size, n_dims_truncated=None, seeds=None):
        if seeds is None:
            # Generate a random seed for each batch to ensure different batches use different seeds
            seeds = torch.randint(0, 2**32, (b_size,))

        xs_b = torch.zeros(b_size, n_points, self.n_dims)
        epsilons_b = torch.zeros_like(xs_b)
        rs_b = torch.zeros(b_size, n_points, 1)
        generator = torch.Generator()

        for i in range(b_size):
            generator.manual_seed(seeds[i].item()) # Initialize the generator with a seed
            # Only generate one point, then replicate to create the remaining points ensuring they are the same within a batch
            x = torch.randn(1, self.n_dims, generator=generator)
            if self.scale is not None:
                x = x @ self.scale
            if self.bias is not None:
                x += self.bias
            xs_b[i] = x.repeat(n_points, 1)  # Replicate the point within the batch
            
            w_epsilons = torch.randn(self.n_dims, self.n_dims, generator=generator)
            epsilons = xs_b[i] @ w_epsilons
            
            #epsilons = torch.randn(n_points, self.n_dims)
            
            rs = torch.rand(n_points, 1)  # Generate a proportion for the perturbation
            epsilons_b[i] = epsilons
            rs_b[i] = rs
            #print(rs)

        if n_dims_truncated is not None:
            xs_b[:, :, n_dims_truncated:] = 0
            epsilons_b[:, :, n_dims_truncated:] = 0

        return xs_b, epsilons_b, rs_b

class FixedGaussianSampler(DataSampler):
    def __init__(self, n_dims, bias=None, scale=None):
        super().__init__(n_dims)
        self.bias = bias
        self.scale = scale

    def sample_xs(self, n_points, b_size, n_dims_truncated=None, seeds=None):
        if seeds is None:
            # Generate a random seed for each batch to ensure different batches use different seeds
            seeds = torch.randint(0, 2**32, (b_size,))

        xs_b = torch.zeros(b_size, n_points, self.n_dims)
        epsilons_b = torch.zeros_like(xs_b)
        rs_b = torch.zeros(b_size, n_points, 1)
        generator = torch.Generator()

        for i in range(b_size):
            generator.manual_seed(seeds[i].item()) # Initialize the generator with a seed
            # Only generate one point, then replicate to create the remaining points ensuring they are the same within a batch
            x = torch.randn(1, self.n_dims, generator=generator)
            if self.scale is not None:
                x = x @ self.scale
            if self.bias is not None:
                x += self.bias
            xs_b[i] = x.repeat(n_points, 1)  # Replicate the point within the batch
            
            w_epsilons = torch.randn(self.n_dims, self.n_dims, generator=generator)
            epsilons = xs_b[i] @ w_epsilons
            # Generate random perturbations for each point
            #epsilons = torch.randn(n_points, self.n_dims)
            
            rs = torch.rand(n_points, 1)  # Generate a proportion for the perturbation
            epsilons_b[i] = epsilons * (1 - rs)
            rs_b[i] = rs

        if n_dims_truncated is not None:
            xs_b[:, :, n_dims_truncated:] = 0
            epsilons_b[:, :, n_dims_truncated:] = 0

        return xs_b, epsilons_b, rs_b


class GaussianSampler(DataSampler):
    def __init__(self, n_dims, bias=None, scale=None):
        super().__init__(n_dims)
        self.bias = bias
        self.scale = scale

    def sample_xs(self, n_points, b_size, n_dims_truncated=None, seeds=None):
        if seeds is None:
            xs_b = torch.randn(b_size, n_points, self.n_dims)
        else:
            xs_b = torch.zeros(b_size, n_points, self.n_dims)
            generator = torch.Generator()
            assert len(seeds) == b_size
            for i, seed in enumerate(seeds):
                generator.manual_seed(seed)
                xs_b[i] = torch.randn(n_points, self.n_dims, generator=generator)
        if self.scale is not None:
            xs_b = xs_b @ self.scale
        if self.bias is not None:
            xs_b += self.bias
        if n_dims_truncated is not None:
            xs_b[:, :, n_dims_truncated:] = 0
        return xs_b
