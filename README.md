# A Theoretical Understanding of Self-Correction through In-context Alignment

This is the official repository for [**A Theoretical Understanding of Self-Correction through
In-context Alignment**](https://arxiv.org/abs/2405.18634) (NeurIPS 2024)

## Environment Setup

To set up the environment, follow these steps:

```bash
conda env create -f environment.yml
conda activate in-context-alignment
```

## Part 1: Synthetic Experiment

To perform synthetic experiments, navigate to the `synthetic_experiments` directory:

```bash
cd synthetic_experiment
```

### Training GPT-2 with Different Heads

To train GPT-2 with different numbers of heads, use the following command, replacing `{head}` with one of the values (1, 3, 4, or 6):

```bash
export IF_FFN=True
export IF_SOFTMAX=True
python train.py --config conf/train-heads-{head}.yaml
```

### Training GPT-2 with Different Layer Counts

To train GPT-2 with different numbers of layers, use the following command, replacing `{n_layer}` with one of the values (5, 10, 15, or 20):

```bash
export IF_FFN=True
export IF_SOFTMAX=True
python train.py --config conf/train-n_layer-{n_layer}.yaml
```

### Training Without FFN or Softmax

To train GPT-2 without Feed-Forward Networks (FFN) or Softmax, use the following command:

```bash
export IF_FFN=False
export IF_SOFTMAX=True
python train.py --config conf/train-n_layer-20.yaml
```

### After training
evaluate the model against the ground truth by running:

```bash
export IF_FFN=True
export IF_SOFTMAX=True
python evaluate.py --config conf/train-n_layer-20.yaml
```

### Experimenting with Noisy Rewards
To conduct experiments with noisy rewards, use the `evaluate_wrong_reward.ipynb` notebook. This notebook allows you to investigate the effects of adding noise to the reward signals during evaluation.

## Part 2: BBQ Evaluation

To perform BBQ evaluation, navigate to the `BBQ` directory:

```bash
cd BBQ
```
To run the BBQ evaluation, use the following command, replacing `{your_save_path}` with your desired save path:

```bash
python -m eval -ds bbq -p question -o {your_save_path}.jsonl --model="llama2"
```

## Part 3: Jailbreaking Evaluation

1. Download the [AdvBench](https://github.com/llm-attacks/llm-attacks) dataset to folder `./data`. Make sure that the path to `harmful_behaviors.csv` is `./data/advbench/harmful_behaviors.csv`.


2. Produce GCG / AutoDAN attack (without defense) on the models to save the adversarial suffixes / prefixes

```
python attack.py --model-path YOUR_MODEL_PATH --save-name YOUR_SAVE_NAME --attack gcg
```

The attack log will be saved to `ATTACK_SAVE_NAME.json` (e.g., vicuna_gcg.json)

3. To run the CaC evaluation, use command 

```
python self_check.py --attack gcg --model-path YOUR_MODEL_PATH --save-name CAC_SAVE_NAME --check-round 1 --fname ATTACK_SAVE_NAME.json
```

If you wish to use history backup for CaC, use command
```
python self_check.py --attack gcg --model-path YOUR_MODEL_PATH --save-name CAC_SAVE_NAME --check-round 1 --fname ATTACK_SAVE_NAME.json --backup
```


## Citation
If you find our work useful, please consider cite our work with
```
@inproceedings{wang2024theoretical,
  title={A Theoretical Understanding of Self-Correction through In-context Alignment},
  author={Wang, Yifei and Wu, Yuyang and Wei, Zeming and Jegelka, Stefanie and Wang, Yisen},
  booktitle={NeurIPS},
  year={2024}
}
```

## Acknowledgement
This repo is partially based upon the following repos:
- https://github.com/dtsip/in-context-learning (Synthetic Experiment for ICL)
- https://github.com/rgambee/self-correction-reproduction (BBQ)
- https://github.com/llm-attacks/llm-attacks (GCG)
- https://github.com/SheltonLiu-N/AutoDAN (AutoDAN)


