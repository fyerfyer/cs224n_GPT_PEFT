# CS 224N Default Final Project: Build GPT-2

This is fyerfyer's implementation of CS 224N's final project. It consists of implementation of basic GPT-2 model and LoRA & ReFT fine tune for downstream task.

Here are some of my implementation notes(continuously updating):

* [GPT-2模型实现笔记1](https://fyerfyer.github.io/posts/gpt2-implementation-notes1/)
* [GPT-2模型实现笔记2](https://fyerfyer.github.io/posts/gpt2-implementation-notes2/)

## Setup instructions

Follow `setup.sh` to properly setup a conda environment and install dependencies.

## Fine-Tuning Scripts

This project includes four fine-tuning scripts for parameter-efficient training:

### LoRA Fine-Tuning

#### 1. LoRA Paraphrase Detection

Train a LoRA-enabled GPT-2 model for paraphrase detection on the Quora dataset:

```bash
python lora_paraphrase_detection.py --use_gpu --epochs 3 --batch_size 12 --lr 1e-4 --lora_rank 8
```

**Key Parameters:**
- `--lora_rank`: LoRA rank for adaptation (default: 8, recommended: 4-16)
- `--lora_alpha`: LoRA scaling parameter (default: 16.0)
- `--lora_dropout`: LoRA dropout rate for regularization (default: 0.1)
- `--early_stopping_patience`: Early stopping patience (default: 5)
- `--batch_size`: Batch size (default: 12, Tesla T4 safe range: 8-16)

#### 2. LoRA Sonnet Generation

Train a LoRA-enabled GPT-2 model for sonnet generation:

```bash
python lora_sonnet_generation.py --use_gpu --epochs 10 --batch_size 4 --lr 1.5e-4 --lora_rank 4
```

**Key Parameters:**
- `--temperature`: Softmax temperature for generation (default: 1.2)
- `--top_p`: Cumulative probability for nucleus sampling (default: 0.9)
- `--early_stopping_patience`: Early stopping patience (default: 5)
- `--min_improvement`: Minimum validation loss improvement (default: 0.01)

### ReFT Fine-Tuning

#### 3. ReFT Paraphrase Detection

Train a ReFT-enabled GPT-2 model for paraphrase detection:

```bash
python reft_paraphrase_detection.py --use_gpu --epochs 3 --batch_size 8 --lr 1e-4 --reft_rank 4
```

**Key Parameters:**
- `--reft_rank`: ReFT intervention rank (default: 4)
- `--reft_alpha`: ReFT scaling parameter (default: 16.0)
- `--reft_activation`: Activation function (default: 'relu', choices: relu, gelu, tanh, linear)
- `--reft_locations`: Intervention locations (default: ['attention'], choices: attention, ffn)
- `--reft_layers`: Specific layers for interventions (default: None for all layers)

#### 4. ReFT Sonnet Generation

Train a ReFT-enabled GPT-2 model for sonnet generation:

```bash
python reft_sonnet_generation.py --use_gpu --epochs 10 --batch_size 4 --lr 1e-4 --reft_rank 4
```

**Key Parameters:**
- `--reft_dropout`: ReFT dropout rate (default: 0.1)
- `--reft_locations`: Where to apply interventions (default: ['attention'])
- `--reft_layers`: Which layers to intervene (default: None for all layers)
- `--temperature`: Generation temperature (default: 1.2)

### Common Parameters

All scripts share these common parameters:
- `--use_gpu`: Enable GPU training
- `--epochs`: Number of training epochs
- `--batch_size`: Training batch size
- `--lr`: Learning rate
- `--model_size`: GPT-2 model size (choices: gpt2, gpt2-medium, gpt2-large)
- `--seed`: Random seed for reproducibility (default: 11711)

### Output Files

- **Paraphrase Detection**: Generates prediction files in `predictions/` directory
- **Sonnet Generation**: Generates sonnets in `predictions/` directory
- **Model Checkpoints**: Saved as `.pt` files with configuration in filename

## Acknowledgement

This project is [CS 224N's default final project](https://web.stanford.edu/class/cs224n/project_w25/CS_224n__Default_Final_Project__Build_GPT_2.pdf).

This project is adapted from a prior year's CS 224N
project [Implement BERT](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1246/project/default-final-project-handout-minbert-spr2024-updated.pdf)
.

Parts of the code are from the [`transformers`](https://github.com/huggingface/transformers)
library ([Apache License 2.0](./LICENSE)).