# LLM-SBM

This repo includes the constructed [alignment data](data/msmarco_dpo.json) and code for SIGIR 2025 submission "Mitigating Source Bias with LLM Alignment".

## Introduction


LLM-SBM (LLM Alignment for Source Bias Mitigation) is a novel framework designed to address source bias in information retrieval systems by aligning LLM-generated outputs with PLM-based retrievers. It introduces an automatic preference data construction pipeline to generate high-quality alignment data and incorporates fine-grained preference differences as weighting factors in the policy training function. This ensures that LLMs produce unbiased outputs without compromising their general capabilities, offering a proactive and scalable solution to enhance the sustainability of the IR ecosystem.


## File Structure
The files in this folder are:
- `data/..`: This folder contains the datasets used for experiments in our experiments. At the beginning, we need to import the data format into the [`data_info.json`](data/dataset_info.json) file. 
- `data_construction/..`: This folder contains the code for our proposed automatic alignment data construction pipeline. 
- `loggers/..`: This folder saves the terminal logs from both training and testing.
- `saves/..`:  This folder includes the checkpoints and results used or saved during training.
- `shell/..`: This folder contains the shell scripts for running LLM-SBM training, model export.
- **`src/llamafactory/`** : This folder contains the code for LLM-SBM training. The specific implementation of LLM-SBM can be found in [this file](src/llamafactory/train/dpo/trainer.py). 
- `yamls/..`: This folder stores the configs corresponding to training, model export.


## A Complete Example

Let's work through a complete example training Llama-3-8B-Instruct.

### Step 1: Set up environment

To facilitate program execution, we follow the environment configuration of LLaMA-Factory, allowing us to directly execute the `llamafactory-cli` command.

```sh
cd LLM-SBM
pip install -e ".[torch,metrics]"
```

### Step 2: Run LLM-SBM
We use LoRA for efficient fine-tuning (for specific configurations, please refer to [this file](yamls/llama3/lora/msmarco_dpo/msmarco_dpo.yaml)). During LLM-SBM training, the trained adapters will be saved under the `saves/`, and the log information will be stored under the `loggers/` directory.

```sh
cd shell
bash LLM-SBM.sh
```
> Note: this command is run on a machine with 1 48GB A6000; on this hardware, LLM-SBM takes about 4hr 40min. 

### Step 3: Export Model for evaluation
Combine the saved adapter with the base model to facilitate future experiments. (for specific configurations, please refer to [this file](yamls/llama3/export/msmarco_dpo/export_msmarco_dpo_2.0.yaml))
```sh
bash export.sh
```

## Key Implementation Details for LLM-SBM:

Owing to their straightforward structures, these loss functions can be implemented concisely within a few lines of code:

```python

# weight: $\delta'$ in the paper.
# self.finetuning_args.weight_alpha: $\alpha'$ in the paper.
# train_eval: only return the LLM-SBM-loss during training.
def preference_aware_dpo_loss(self, policy_chosen_logps, policy_rejected_logps, reference_chosen_logps, reference_rejected_logps, train_eval, weight, description = "Add the preference weights to the original dpo losses."
    ):

        ori_losses, chosen_rewards, rejected_rewards = self.dpo_loss(
                    policy_chosen_logps, policy_rejected_logps, reference_chosen_logps, reference_rejected_logps
                )
        
        weight = weight[:len(weight)//2]
        weight_losses = torch.pow(weight, self.finetuning_args.weight_alpha) * ori_losses
        if train_eval == "train": 
            return weight_losses, chosen_rewards, rejected_rewards
        else:
            return ori_losses, chosen_rewards, rejected_rewards
```

## Construct Your Custom Alignment Data
Please see our [code](data_construction/preference_data.py) for details. You can construct your custom alignment data with other corpus or other paramater settings.

## Evaluation of Source Bias
For evaluation of source bias, we use the official code from the [Cocktail](https://github.com/KID-22/Cocktail) benchmark.


## Acknowledge
LLM-SBM is built based on the following project: [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory). Note that we only retain the parts of this framework that were useful to LLM-SBM and make adaptive adjustments to it.