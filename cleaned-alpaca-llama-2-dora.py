# This is a modified version of TRL's `SFTTrainer` example (https://github.com/huggingface/trl/blob/main/examples/scripts/sft_trainer.py),
# adapted to run with DeepSpeed ZeRO-3 and Llama2-7B or Mistral-7B The settings below were run on 1 node of 4 x A10 (24GB) GPUs.

# Usage:
#   - Install the latest transformers & accelerate versions: `pip install -U transformers accelerate`
#   - Install deepspeed: `pip install deepspeed==0.13.1`
#   - Install TRL from main: pip install git+https://github.com/huggingface/trl.git
#   - Clone the repo: git clone github.com/huggingface/trl.git
#   - Copy this Gist into trl/examples/scripts
#   - Run from root of trl repo with: accelerate launch --config_file=examples/accelerate_configs/deepspeed_zero1.yaml examples/scripts/cleaned-alpaca-llama-2-dora.py

from accelerate import Accelerator
from alpaca_utils import Prompter
from datasets import load_dataset
from dataclasses import dataclass, field

import os
from peft import LoraConfig

import torch
from trl import SFTTrainer
from transformers import (
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    AutoTokenizer
)
from typing import Optional

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = "1"

prompter = Prompter("alpaca")


# Define and Parse Arguments using HF Parser
@dataclass
class ScriptArguments:
    """
    The name of the Casual LM model we wish to finetune with SFTTrainer
    """

    model_name: Optional[str] = field(default="meta-llama/Llama-2-7b-hf", metadata={"help": "the model name"})
    dataset_text_field: Optional[str] = field(default="instruction", metadata={"help": "the text field of the dataset"})
    log_with: Optional[str] = field(default="wandb", metadata={"help": "use 'wandb' to log with wandb"})
    learning_rate: Optional[float] = field(default=4.0e-4, metadata={"help": "the learning rate"})
    batch_size: Optional[int] = field(default=4, metadata={"help": "the batch size"})
    seq_length: Optional[int] = field(default=1024, metadata={"help": "Input sequence length"})
    gradient_accumulation_steps: Optional[int] = field(
        default=4,
        metadata={"help": "the number of gradient accumulation steps. should match with deep speed config file"}
    )
    load_in_4bit: Optional[bool] = field(default=True, metadata={"help": "load the model in 4 bits precision"})
    use_peft: Optional[bool] = field(default=True, metadata={"help": "Whether to use PEFT or not to train adapters"})
    output_dir: Optional[str] = field(default="output/", metadata={"help": "the output directory"})
    peft_lora_r: Optional[int] = field(default=64, metadata={"help": "the r parameter of the LoRA adapters"})
    peft_lora_alpha: Optional[int] = field(default=128, metadata={"help": "the alpha parameter of the LoRA adapters"})
    logging_steps: Optional[int] = field(default=50, metadata={"help": "the number of logging steps"})
    num_train_epochs: Optional[int] = field(default=1, metadata={"help": "the number of training epochs"})
    max_steps: Optional[int] = field(default=-1, metadata={"help": "the number of training steps"})
    save_steps: Optional[int] = field(default=100,
                                      metadata={"help": "Number of updates steps before two checkpoint saves"})


parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

# Step 1: Load the tokenizer and set padding token to 0
tokenizer = AutoTokenizer.from_pretrained(script_args.model_name, use_fast=True)
tokenizer.pad_token_id = 0
tokenizer.padding_side = "right"


def tokenize(prompt, add_eos_token=True):
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=512,
        padding=False,
        return_tensors=None,
    )
    if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < 512
            and add_eos_token
    ):
        result["input_ids"].append(tokenizer.eos_token_id)
        result["attention_mask"].append(1)

    result["labels"] = result["input_ids"].copy()

    return result


def generate_and_tokenize_prompt(data_point):
    full_prompt = prompter.generate_prompt(
        data_point["instruction"],
        data_point["input"],
        data_point["output"],
    )
    tokenized_full_prompt = tokenize(full_prompt)
    return tokenized_full_prompt


# Step 2: Load dataset from examples/ folder of TRL
data_path = "examples/alpaca_data_cleaned_archive.json"
if data_path.endswith(".json") or data_path.endswith(".jsonl"):
    data = load_dataset("json", data_files=data_path)
else:
    data = load_dataset(data_path)

# Step 3: Split the data into Train and Test sets
train_val = data["train"].train_test_split(
    test_size=2000, shuffle=True, seed=42
)
train_data = (
    train_val["train"].shuffle().map(generate_and_tokenize_prompt)
)
val_data = (
    train_val["test"].shuffle().map(generate_and_tokenize_prompt)
)

print(f"Training Data : {train_data}")
print(f"Validation Data : {val_data}")

# Step 4: Load the model with 4-bit Quantization
compute_dtype = getattr(torch, "float16")
if script_args.load_in_4bit:
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=False,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
    )
    # Copy the model to each GPU device
    device_map = {"": Accelerator().local_process_index}
    torch_dtype = torch.bfloat16
else:
    device_map = None
    quantization_config = None
    torch_dtype = None

model = AutoModelForCausalLM.from_pretrained(
    script_args.model_name,
    quantization_config=quantization_config,
    device_map=device_map,
    trust_remote_code=script_args.trust_remote_code,
    torch_dtype=torch_dtype,
)
model.config.use_cache = False
model.config.pretraining_tp = 1


# Step 5: Define the DoRA Config. I'm using the below hyperparameters as specified in the research paper
if script_args.use_peft:
    peft_config = LoraConfig(
        use_dora=True,
        r=script_args.peft_lora_r,
        lora_alpha=script_args.peft_lora_alpha,
        lora_dropout=0.0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM",
        inference_mode=False
    )
else:
    peft_config = None


# Step 6: Define the training arguments
training_arguments = TrainingArguments(
    bf16=True,
    warmup_ratio=0.1,
    eval_steps=100,
    evaluation_strategy='steps',
    lr_scheduler_type="cosine",
    gradient_checkpointing=True,
    save_steps=script_args.save_steps,
    logging_steps=script_args.logging_steps,
    learning_rate=script_args.learning_rate,
    max_steps=script_args.max_steps,
    output_dir=script_args.output_dir,
    num_train_epochs=script_args.num_train_epochs,
    per_device_train_batch_size=script_args.batch_size,
    gradient_accumulation_steps=script_args.gradient_accumulation_steps,
)

# Step 7: Define the Trainer
trainer = SFTTrainer(
    model=model,
    packing=False,
    args=training_arguments,
    train_dataset=train_data,
    eval_dataset=val_data,
    peft_config=peft_config,
    max_seq_length=script_args.seq_length,
    tokenizer=tokenizer,
    dataset_text_field=script_args.dataset_text_field,
)

trainer.train()

# Step 8: Save the model
trainer.save_model(script_args.output_dir)
