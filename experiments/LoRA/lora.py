import modal
from datetime import datetime
from random import uniform

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install(
        "accelerate",
        "transformers",
        "torch",
        "datasets",
        "huggingface_hub",
        "trl",
        "peft",
        "bitsandbytes",
        "wandb",
    )
    .env(dict())
    .entrypoint([])
)

app = modal.App(
    "lm-as-memory-LoRA",
    image=image,
)

vol = modal.Volume.from_name(
    "lm-as-memory",
    create_if_missing=True,
)

model_name = "microsoft/phi-4"  # 14.7B / 27GB

dataset_path = "HuggingFaceTB/smoltalk"
dataset_name = "everyday-conversations"

cache_dir = "./.cache/huggingface"

import time


def est_time():
    # Eastern Standard Time is UTC-5 hours
    return time.time() - 5 * 60 * 60


run_name = f"./runs/{model_name.replace('/', '_')}--{dataset_path.replace('/', '_')}--{datetime.fromtimestamp(est_time()).strftime('%Y%m%d-%H%M')}"

default_inputs = [
    {
        "role": "LA",
        "content": "Which newspaper did Etel Adnan work at in Beirut? Hint, Etel Adnan mentioned this in her interview with Laure Adler.",
    },
    {
        "role": "LA",
        "content": "Which newspaper did Etel Adnan work at in Beirut? Hint, Etel Adnan mentioned this in her interview with Laure Adler.",
    },
    {
        "role": "LA",
        "content": "Which newspaper did Etel Adnan work at in Beirut? Hint, Etel Adnan mentioned this in her interview with Laure Adler.",
    },
    {
        "role": "LA",
        "content": "Which newspaper did Etel Adnan work at in Beirut? Hint, Etel Adnan mentioned this in her interview with Laure Adler.",
    },
    {
        "role": "LA",
        "content": "Which newspaper did Etel Adnan work at in Beirut? Hint, Etel Adnan mentioned this in her interview with Laure Adler.",
    },
]  # correct answer = L’Orient-Le Jour.


def generate_and_print(
    model, tokenizer, device, inputs, max_new_tokens, generation_prompt
):
    templated_inputs = [
        tokenizer.apply_chat_template(
            [input],
            tokenize=False,
            generation_prompt=generation_prompt,
        )
        for input in inputs
    ]

    # Batch tokenize
    batch_inputs = tokenizer(
        templated_inputs,
        return_tensors="pt",
        padding=True,
        padding_side="left",
    ).to(device)

    # Batch generate
    print("Generating...")
    outputs = model.generate(
        **batch_inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=uniform(0.7, 1.5),  # Randomize temperature for exploration
        top_p=uniform(0.85, 1.0),  # Randomize nucleus sampling parameter
        top_k=int(uniform(40, 100)),  # Randomize top-k value
        pad_token_id=tokenizer.pad_token_id,
    )
    print("Done")

    # Decode all outputs
    responses = [
        tokenizer.decode(output, skip_special_tokens=False) for output in outputs
    ]
    for response in responses:
        print(response)
        print("\n----------------\n")


@app.function(
    image=image,
    volumes={"/data": vol},
    secrets=[modal.Secret.from_name("my-huggingface-secret")],
    timeout=30 * 60,
)
def load_tokenizer_and_model():
    from huggingface_hub import snapshot_download
    from datasets import load_dataset
    import os

    os.chdir("/data")

    snapshot_download(
        repo_id=model_name,
        cache_dir=cache_dir,
    )
    vol.commit()

    load_dataset(
        dataset_path,
        name=dataset_name,
        cache_dir=cache_dir,
    )
    vol.commit()


@app.function(
    image=image,
    gpu="T4",  # A10g=24GB(1.1$/h) / L4=24GB(0.8$/h) / T4=16GB(0.59$/h)
    volumes={"/data": vol},
    timeout=1 * 60 * 60,
)
def generate_with_base_model():
    import os
    from random import uniform
    import torch
    from trl import setup_chat_format
    from transformers import (
        BitsAndBytesConfig,
        AutoModelForCausalLM,
        AutoTokenizer,
        StoppingCriteria,
        StoppingCriteriaList,
    )

    class StopOnEndToken(StoppingCriteria):
        def __init__(self, end_token_id):
            self.end_token_id = end_token_id

        def __call__(self, input_ids, scores):
            if input_ids[0][-1] == self.end_token_id:
                return True
            return False

    os.chdir("/data")

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print("device:", device)

    config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        quantization_config=config,
        local_files_only=True,
    ).to(device)

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        local_files_only=True,
    )

    print("Generating with ", model_name)
    generate_and_print(model, tokenizer, device, default_inputs)


@app.function(
    image=image,
    gpu="A10g",
    volumes={"/data": vol},
    timeout=1 * 60 * 60,
    secrets=[modal.Secret.from_name("wandb-secret")],
)
def fine_tune_with_lora(train_dataset):
    import os
    import torch
    from datasets import load_dataset, Dataset
    from transformers import (
        BitsAndBytesConfig,
        AutoModelForCausalLM,
        AutoTokenizer,
        DataCollatorWithPadding,
        DefaultDataCollator,
    )
    from trl import SFTTrainer, setup_chat_format, SFTConfig
    from peft import LoraConfig
    from torch.utils.data import DataLoader
    import wandb

    os.chdir("/data")

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print("device:", device)

    config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name, cache_dir=cache_dir, quantization_config=config
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    tokenizer.chat_template = "{% for message in messages %}{% if (message['role'] == 'system') %}{{'<|im_start|>system<|im_sep|>' + message['content'] + '<|im_end|>'}}{% elif (message['role'] == 'LA') %}{{'<|im_start|>&29njkn(dkj38$%nkjn#<|im_sep|>' + message['content'] + '<|im_end|>'}}{% elif (message['role'] == 'EA') %}{{'<|im_start|>foi%ioh!@oih(&idl*<|im_sep|>' + message['content'] + '<|im_end|>'}}{% elif (message['role'] == 'user') %}{{'<|im_start|>user<|im_sep|>' + message['content'] + '<|im_end|>'}}{% elif (message['role'] == 'assistant') %}{{'<|im_start|>assistant<|im_sep|>' + message['content'] + '<|im_end|>'}}{% endif %}{% endfor %}"

    role_A = "#29njkn(dkj38$%nkjn#"  # Laure Adler
    role_B = "#foi*Ewoh!@oih(&idl#"  # Etel Adnan
    print("vocab length before adding special tokens:", len(tokenizer))
    tokenizer.add_special_tokens(
        {
            "additional_special_tokens": tokenizer.additional_special_tokens
            + [role_A, role_B, "<|im_sep|>"]
        }
    )
    tokenizer.save_pretrained(run_name + "/tokenizer")

    print("resize token embeddings with new vocab length: ", len(tokenizer))
    model.resize_token_embeddings(
        len(tokenizer),
        mean_resizing=False,  # New tokens are initialized with random values instead of mean
    )

    tokenizer.pad_token = tokenizer.eos_token

    if train_dataset is None:
        dataset = load_dataset(dataset_path, name=dataset_name, cache_dir=cache_dir)
    else:
        dataset = {"train": Dataset.from_dict(train_dataset)}

    # Configure LoRA parameters
    rank_dimension = 12
    lora_alpha = 8
    lora_dropout = 0.05

    peft_config = LoraConfig(
        r=rank_dimension,  # Rank dimension - typically between 4-32
        lora_alpha=lora_alpha,  # LoRA scaling factor - typically 2x rank
        lora_dropout=lora_dropout,  # Dropout probability for LoRA layers
        bias="none",  # Bias type for LoRA. the corresponding biases will be updated during training.
        target_modules="all-linear",  # Which modules to apply LoRA to
        task_type="CAUSAL_LM",  # Task type for model architecture
    )

    num_train_epochs = 10
    batch_size = 1
    gradient_accumulation_steps = 1
    learning_rate = 2e-4  # Learning rate (QLoRA paper)
    max_grad_norm = 0.3  # Gradient clipping threshold for AdamW
    warmup_ratio = 0.0
    lr_scheduler_type = "constant"  # Keep learning rate constant after warmup
    max_seq_length = 4096

    # Initialize wandb
    wandb.init(
        project="lm-as-memory",
        name=run_name,
        config={
            "model": model_name,
            "dataset": dataset_path,
            "rank_dimension": rank_dimension,
            "lora_alpha": lora_alpha,
            "lora_dropout": lora_dropout,
            "learning_rate": learning_rate,
            "num_train_epochs": num_train_epochs,
            "batch_size": batch_size,
        },
    )

    # Training configuration
    # Hyperparameters based on QLoRA paper recommendations
    args = SFTConfig(
        output_dir=run_name,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,  # Accumulate gradients for larger effective batch
        gradient_checkpointing=True,  # Trade compute for memory savings
        optim="adamw_torch_fused",
        learning_rate=learning_rate,
        max_grad_norm=max_grad_norm,
        warmup_ratio=warmup_ratio,
        lr_scheduler_type=lr_scheduler_type,
        logging_strategy="epoch",
        save_strategy="epoch",
        bf16=True,
        push_to_hub=False,
        report_to="wandb",
        max_seq_length=max_seq_length,
        packing=False,  # Don't concatenate multiple sequences to meet max_seq_length
    )

    # Create SFTTrainer with LoRA configuration
    trainer = SFTTrainer(
        model=model,
        args=args,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        train_dataset=dataset["train"],
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    trainer.train()
    trainer.save_model(output_dir=run_name)

    # Close wandb run
    wandb.finish()


@app.function(
    image=image,
    volumes={"/data": vol},
    gpu="A10g",
    timeout=10 * 60,
)
def generate_with_finetuned_model():
    from peft import AutoPeftModelForCausalLM, PeftConfig
    from transformers import AutoTokenizer, BitsAndBytesConfig
    import torch
    import os

    os.chdir("/data")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    lora_model_name = "./runs/microsoft_phi-4--HuggingFaceTB_smoltalk--20250124-1830"

    tokenizer = AutoTokenizer.from_pretrained(
        lora_model_name,
        cache_dir=cache_dir,
        local_files_only=True,
    )
    role_A = "#29njkn(dkj38$%nkjn#"  # Laure Adler
    role_B = "#foi*Ewoh!@oih(&idl#"  # Etel Adnan
    tokenizer.add_special_tokens(
        {
            "additional_special_tokens": tokenizer.additional_special_tokens
            + [role_A, role_B, "<|im_sep|>"]
        }
    )

    checkpoint_steps = [48, 96, 160]
    # checkpoint_steps = [16, 32, 48, 64, 80, 96, 112, 128, 144, 160]

    for step in checkpoint_steps:
        print(f">>>>>>> step {step} >>>>>>>")
        model = AutoPeftModelForCausalLM.from_pretrained(
            (
                lora_model_name
                + ("" if step == checkpoint_steps[-1] else f"/checkpoint-{step}")
            ),
            cache_dir=cache_dir,
            quantization_config=quantization_config,
            local_files_only=True,
        ).to(device)

        # print("Generating with ", lora_model_name)
        generate_and_print(
            model,
            tokenizer,
            device,
            inputs=[
                {
                    "role": "system",
                    "content": "Explain which newspaper did Etel Adnan work at in Beirut?",
                },
                {"role": "LA", "content": "Which newspaper did you work at in Beirut?"},
                {"role": "system", "content": "Greet the user first"},
            ],
            max_new_tokens=50,
            generation_prompt=False,
        )

        # remove the model from memory
        del model


@app.local_entrypoint()
def main():
    import json

    print("local entrypoint")

    # load_tokenizer_and_model.remote()
    # print("loaded tokenizer and model")

    # generate_with_base_model.remote()
    # print("generated with base model")

    # with open("./datasets/etel_adnan_tokens_with_labels.json", "r") as f:
    #     data = f.read()
    #     data = json.loads(data)

    # fine_tune_with_lora.remote(data)
    # print("fine-tuned with LoRA")

    generate_with_finetuned_model.remote()
    print("generated with finetuned model")
