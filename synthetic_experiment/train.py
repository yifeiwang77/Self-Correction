import os
from random import randint
import uuid

from quinine import QuinineArgumentParser
from tqdm import tqdm
import torch
import yaml

from tasks import get_task_sampler, mse_loss
from samplers import get_data_sampler
from curriculum import Curriculum
from schema import schema
from models import build_model, build_NoSoftmax_model, build_NoFFN_model
import wandb
import matplotlib.pyplot as plt
import numpy as np
import argparse
torch.backends.cudnn.benchmark = True


def train_step(model, xs, ini_ys, ys, rs, optimizer, loss_func):
    optimizer.zero_grad()
    output = model(xs, ys, rs)

    loss = loss_func(output, ys, rs, ini_ys)
    loss.backward()
    optimizer.step()
    return loss.detach().item(), output.detach()


def sample_seeds(total_seeds, count):
    seeds = set()
    while len(seeds) < count:
        seeds.add(randint(0, total_seeds - 1))
    return seeds


def train(model, args):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.training.learning_rate )
    curriculum = Curriculum(args.training.curriculum)

    starting_step = 0
    state_path = os.path.join(args.out_dir, "state.pt")
    if os.path.exists(state_path):
        state = torch.load(state_path)
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        starting_step = state["train_step"]
        for i in range(state["train_step"] + 1):
            curriculum.update()

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
    pbar = tqdm(range(starting_step, args.training.train_steps))

    num_training_examples = args.training.num_training_examples

    for i in pbar:
        data_sampler_args = {}
        task_sampler_args = {}

        if "sparse" in args.training.task:
            task_sampler_args["valid_coords"] = curriculum.n_dims_truncated
        if num_training_examples is not None:
            assert num_training_examples >= bsize
            seeds = sample_seeds(num_training_examples, bsize)
            data_sampler_args["seeds"] = seeds
            task_sampler_args["seeds"] = [s + 1 for s in seeds]

        xs, epsilons, rs = data_sampler.sample_xs(
            curriculum.n_points,
            bsize,
            curriculum.n_dims_truncated,
            **data_sampler_args,
        )
        
        task = task_sampler(**task_sampler_args)
        ini_ys, ys = task.evaluate(xs, epsilons, rs) # save ys for eval

        loss_func = task.get_training_metric()

        loss, output = train_step(model, xs.cuda(), ini_ys.cuda(), ys.cuda(), rs.cuda(), optimizer, loss_func)

        eval_losses = torch.cosine_similarity(output.cuda(), ini_ys.cuda(), dim=2).mean(dim=0).detach()

        cum_sum = torch.cumsum(ys, dim=1)

        b, num, dim = ys.size()
        divisors = torch.arange(1, num + 1).view(1, num, 1).expand(b, num, dim).type_as(ys)

        mean_ys_b = cum_sum / divisors
        baseline_losses = mse_loss(mean_ys_b.cuda(), ini_ys.cuda()).mean(dim=2).mean(dim=0).detach()  

        if i % args.wandb.log_every_steps == 0 and not args.test_run:
            wandb.log(
                {
                    "baseline_loss": baseline_losses[-1],
                    "eval_loss_0": eval_losses[0],
                    "eval_loss_final": eval_losses[-1],
                    "overall_loss": loss,
                    "n_points": curriculum.n_points,
                    "n_dims": curriculum.n_dims_truncated,
                },
                step=i,
            )

        curriculum.update()

        pbar.set_description(f"loss {loss}")
        if i % args.training.save_every_steps == 0 and not args.test_run:
            training_state = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_step": i,
            }
            torch.save(training_state, state_path)

        if (
            args.training.keep_every_steps > 0
            and i % args.training.keep_every_steps == 0
            and not args.test_run
            and i > 0
        ):
            torch.save(model.state_dict(), os.path.join(args.out_dir, f"model_{i}.pt"))


def main(args, use_FFN, use_softmax):
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
    
    model.cuda()
    model.train()
    train(model, args)



if __name__ == "__main__":
    parser = QuinineArgumentParser(schema=schema)
    args = parser.parse_quinfig()
    use_FFN = os.getenv('IF_FFN', 'True').lower() in ('true', '1', 't')
    use_softmax = os.getenv('IF_SOFTMAX', 'True').lower() in ('true', '1', 't')
    print(f"Use FFN: {use_FFN}")
    print(f"Use Softmax: {use_softmax}")

    assert use_FFN or use_softmax 
    
    if not args.test_run:
        out_dir = os.path.join(args.out_dir, f'FFN_{use_FFN}_Softmax_{use_softmax}')
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        args.out_dir = out_dir

        with open(os.path.join(out_dir, "config.yaml"), "w") as yaml_file:
            yaml.dump(args.__dict__, yaml_file, default_flow_style=False)
    
    main(args, use_FFN, use_softmax)
