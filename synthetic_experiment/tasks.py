import math

import torch


def squared_error(ys_pred, ys):
    return (ys - ys_pred).square()


def mean_squared_error(ys_pred, ys):
    return (ys - ys_pred).square().mean()


def accuracy(ys_pred, ys):
    return (ys == ys_pred.sign()).float()


sigmoid = torch.nn.Sigmoid()
bce_loss = torch.nn.BCELoss()
mse_loss = torch.nn.MSELoss(reduction='none')

def cross_entropy(ys_pred, ys):
    output = sigmoid(ys_pred)
    target = (ys + 1) / 2 
    return bce_loss(output, target)

#return torch.cosine_similarity(y_pred.squeeze(0) , ys, dim=1)
def r_theta(y_pred, ys):
    # y_pred_expanded = y_pred.unsqueeze(0).expand(ys.size(0), -1)
    # t = mse_loss(y_pred_expanded, ys)
    # a = -t.mean(dim=1)
    # return a
    return torch.cosine_similarity(y_pred.squeeze(0) , ys, dim=1)

def pl_loss_single(y_pred, ys, rs):
    
    _, indices = torch.sort(rs.squeeze(), descending=True)
    sorted_ys = ys[indices]
    #print("sorted_ys",sorted_ys)
    N = len(sorted_ys)
    
    # Precompute all cosine similarities in one go to leverage vectorization
    similarities = r_theta(y_pred, sorted_ys)
    #print(y_pred.shape,sorted_ys.shape)
    #print(similarities)
    exp_similarities = torch.exp(similarities)
    #print(exp_similarities)
    # Compute the denominator once for all i by cumulatively summing exp similarities in reverse
    cumsum_reverse_exp_similarities = torch.flip(torch.cumsum(torch.flip(exp_similarities, dims=[0]), dim=0), dims=[0])
    #print("cumsum:",cumsum_reverse_exp_similarities)
    loss_elements = -torch.log(exp_similarities / cumsum_reverse_exp_similarities)
    #print(loss_elements)
    #loss = torch.sum(loss_elements)
    #print(loss_elements)
    loss = loss_elements.sum()
    #loss = loss_elements[0]
    return loss #control the /\ Wx-y/| 

def pl_loss_mean_batch(ys_pred_b, ys_b, rs_b, ini_ys_b):
    total_loss = 0
    total_count = 0
    for batch_index in range(len(ys_pred_b)):
        ys_pred = ys_pred_b[batch_index]
        ys = ys_b[batch_index]
        rs = rs_b[batch_index]
        batch_loss = 0
        count = 0
        for n, y_pred in enumerate(ys_pred, start=0):  
            if n < 2:
                continue # Ensuring to start from 2
            current_ys = ys[:n]
            current_rs = rs[:n]
            #print(current_ys.shape)
            if len(current_ys) > 1:
                batch_loss += pl_loss_single(y_pred, current_ys, current_rs)
                count += 1
                
        if count > 0:
            total_loss += batch_loss / count
            total_count += 1
    
    average_loss = total_loss / total_count if total_count > 0 else 0
    return average_loss



class Task:
    def __init__(self, n_dims, batch_size, pool_dict=None, seeds=None):
        self.n_dims = n_dims
        self.b_size = batch_size
        self.pool_dict = pool_dict
        self.seeds = seeds
        assert pool_dict is None or seeds is None

    def evaluate(self, xs):
        raise NotImplementedError

    @staticmethod
    def generate_pool_dict(n_dims, num_tasks):
        raise NotImplementedError

    @staticmethod
    def get_metric():
        raise NotImplementedError

    @staticmethod
    def get_training_metric():
        raise NotImplementedError


def get_task_sampler(
    task_name, n_dims, batch_size, pool_dict=None, num_tasks=None, **kwargs
):
    task_names_to_classes = {
        "mixed_linear_alignment": MixedLinearAlignment,
    }
    if task_name in task_names_to_classes:
        task_cls = task_names_to_classes[task_name]
        if num_tasks is not None:
            if pool_dict is not None:
                raise ValueError("Either pool_dict or num_tasks should be None.")
            pool_dict = task_cls.generate_pool_dict(n_dims, num_tasks, **kwargs)
        return lambda **args: task_cls(n_dims, batch_size, pool_dict, **args, **kwargs)
    else:
        print("Unknown task")
        raise NotImplementedError
        
        

class MixedLinearAlignment(Task):
    def __init__(self, n_dims, batch_size, pool_dict=None, seeds=None, scale=1):
        """scale: a constant by which to scale the randomly sampled weights."""
        super(MixedLinearAlignment, self).__init__(n_dims, batch_size, pool_dict, seeds)
        self.scale = scale
        if pool_dict is None and seeds is None:
            self.w_b = torch.randn(self.b_size, self.n_dims, self.n_dims)
            #w = torch.randn(self.n_dims, self.n_dims)
            #self.w_b = w.repeat(self.b_size, 1, 1)
        elif seeds is not None:
            self.w_b = torch.zeros(self.b_size, self.n_dims, self.n_dims)
            generator = torch.Generator()
            assert len(seeds) == self.b_size
            for i, seed in enumerate(seeds):
                generator.manual_seed(seed)
                self.w_b[i] = torch.randn(self.n_dims, self.n_dims, generator=generator)
        else:
            assert "w" in pool_dict
            indices = torch.randperm(len(pool_dict["w"]))[:batch_size]
            self.w_b = pool_dict["w"][indices]

    def evaluate(self, xs_b, epsilon_b, rs_b):
        w_b = self.w_b.to(xs_b.device)
        
        ini_ys_b = self.scale * (xs_b @ w_b)
        ys_b = ini_ys_b * rs_b + epsilon_b * (1 - rs_b)
        #ys_b = ini_ys_b  + epsilon_b * (1 - rs_b)
        return ini_ys_b, ys_b

    @staticmethod
    def generate_pool_dict(n_dims, num_tasks, **kwargs):  # ignore extra args
        return {"w": torch.randn(num_tasks, n_dims, 1)}

    @staticmethod
    def get_eval_metric():
        return mse_loss

    @staticmethod
    def get_training_metric():
        return pl_loss_mean_batch
    


def mse_for_align(ys_pred_b, ys_b, rs_b, ini_ys_b):
    tmp_mse = torch.nn.MSELoss(reduction='mean')
    a = tmp_mse(ys_pred_b, ini_ys_b).mean()
    return a



