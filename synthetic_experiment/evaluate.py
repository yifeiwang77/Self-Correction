import os
from random import randint
import uuid

from quinine import QuinineArgumentParser
from tqdm import tqdm
import torch
import yaml
from tasks import get_task_sampler
from samplers import get_data_sampler
from schema import schema
from models import build_model, build_NoSoftmax_model, build_NoFFN_model, build_GD_model
from tasks import pl_loss_single
import wandb
import numpy as np
torch.backends.cudnn.benchmark = True

def eval_step(model, xs, ini_ys, ys, rs, loss_func):
    model.eval()
    output = model(xs, ys, rs)
    loss = loss_func(output, ys, rs, ini_ys)
    return loss.detach().item(), output.detach()


def sample_seeds(total_seeds, count):
    seeds = set()
    while len(seeds) < count:
        seeds.add(randint(0, total_seeds - 1))
    return seeds


def evaluate(model, args):
    n_dims = model.n_dims
    bsize = args.training.batch_size
    data_sampler = get_data_sampler(args.training.data, n_dims=n_dims)
    task_sampler = get_task_sampler(
        args.training.task,
        n_dims,
        bsize,
        num_tasks=args.training.num_tasks,
        **args.training.task_kwargs,
    )

    num_training_examples = args.training.num_training_examples

    data_sampler_args = {}
    task_sampler_args = {}

    if num_training_examples is not None:
        assert num_training_examples >= bsize
        seeds = sample_seeds(num_training_examples, bsize)
        data_sampler_args["seeds"] = seeds
        task_sampler_args["seeds"] = [s + 1 for s in seeds]

    xs, epsilons, rs = data_sampler.sample_xs(
        args.model.n_positions,
        bsize,
        n_dims,
        **data_sampler_args,
    )
    
    task = task_sampler(**task_sampler_args)
    ini_ys, ys = task.evaluate(xs, epsilons, rs) # save ys for eval
    
    loss_func = task.get_training_metric()

    loss, output = eval_step(model, xs.cuda(), ini_ys.cuda(), ys.cuda(), rs.cuda(), loss_func)


    eval_losses = 1 - torch.cosine_similarity(output.cuda(), ini_ys.cuda(), dim=2).mean(dim=0).detach()
    cum_sum = torch.cumsum(ys, dim=1)

    b, num, dim = ys.size()
    divisors = torch.arange(1, num + 1).view(1, num, 1).expand(b, num, dim).type_as(ys)

    mean_ys_b = cum_sum / divisors
    baseline_losses = torch.cosine_similarity(mean_ys_b.cuda(), ini_ys.cuda(), dim=2).mean(dim=0).detach()
    



    baseline_losses = baseline_losses.cpu().numpy()
    eval_losses = eval_losses.cpu().numpy()
    

    
    GD_eval_sum = np.zeros(num)
    for batch in tqdm(range(b)):
        y_tmp = ys[batch]
        r_tmp = rs[batch]
        GD_eval = np.zeros(num)
        for i in range(num):
            if i>0:
                model_GD = build_GD_model(args.model).cuda().train()
                tmp_optimizer = torch.optim.Adam(model_GD.parameters(), lr=0.1)
                n=50
                for step in range(n):
                    tmp_optimizer.zero_grad()
                    GD_output = model_GD(xs[batch][0].cuda())
                    
                    current_ys = y_tmp[:i+1]
                    current_rs = r_tmp[:i+1]
                    tmploss = pl_loss_single(GD_output.cuda(), current_ys.cuda(), current_rs.cuda())
                    tmploss.backward()
                    tmp_optimizer.step()


                GD_output_new = model_GD(xs[batch][0].cuda())
                cos_loss = torch.cosine_similarity(GD_output_new.squeeze().cuda(), ini_ys[batch].cuda()).detach()
                GD_eval[i] = 1 - cos_loss[0]
            else:
                GD_eval[0] = 0
        GD_eval_sum += GD_eval
        
    GD_eval_avg = GD_eval_sum / b
    t_values = list(range(2, 16))
    # We have one position shift here because the prediction of transformer in position N only uses [0:N-1] xyrs as information.
    print(eval_losses[2:16])
    print(GD_eval_avg[1:15])

def main(args):
    if args.test_run:
        curriculum_args = args.training.curriculum
        curriculum_args.points.start = curriculum_args.points.end
        curriculum_args.dims.start = curriculum_args.dims.end
        args.training.train_steps = 100
    else:
        wandb.init(
            dir=args.out_dir,
            project=args.wandb.project,
            entity=args.wandb.entity,
            config=args.__dict__,
            notes=args.wandb.notes,
            name=args.wandb.name,
            resume=True,
        )
    
    model= build_model(args.model)
    if not use_FFN:
        model = build_NoFFN_model(args.model)
    if not use_softmax:
        model = build_NoSoftmax_model(args.model)
    state_dict = torch.load(os.path.join(args.out_dir,'state.pt'))
    model.load_state_dict(state_dict["model_state_dict"])
    model.cuda()
    model.eval()
    evaluate(model, args)


if __name__ == "__main__":
    use_FFN = os.getenv('IF_FFN', 'True').lower() in ('true', '1', 't')
    use_softmax = os.getenv('IF_SOFTMAX', 'True').lower() in ('true', '1', 't')
    assert use_FFN or use_softmax 
    parser = QuinineArgumentParser(schema=schema)
    args = parser.parse_quinfig()
    print(f"Running with: {args}")
    

    out_dir = os.path.join(args.out_dir, f'FFN_{use_FFN}_Softmax_{use_softmax}')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    args.out_dir = out_dir

    with open(os.path.join(out_dir, "config.yaml"), "w") as yaml_file:
        yaml.dump(args.__dict__, yaml_file, default_flow_style=False)

    main(args)
