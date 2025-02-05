import time
import modal
from datetime import datetime

from transformers.utils import quantization_config
from utils.generate import generate_and_print

model_name = "microsoft/phi-4"  # 14.7B / 27GB
dataset_path = "./dataset"  # 27 samples with 4096 tokens
cache_dir = "./.cache/huggingface"
run_name = f"{model_name.split('/')[-1]}_ContinuedPretraining_{datetime.fromtimestamp(time.time() - 5 * 60 * 60).strftime('%m%d-%H:%M')}"
output_dir = f"./runs/{run_name}"

# Training Config
num_train_epochs = 1
batch_size = 1
optimizer = "adamw_torch_fused"
gradient_accumulation_steps = 1
learning_rate = 2e-4  # Learning rate from QLoRA paper
max_grad_norm = 0.3  # Gradient clipping threshold for AdamW
warmup_ratio = 0.03  # Need for AdamW to gather the moving average.
lr_scheduler_type = "constant"  # Keep learning rate constant after warmup
max_seq_length = 256
gradient_checkpointing = True

# Modal Fine-Tuning Config
number_of_gpu = 4
modal_fine_tuning_gpu = f"a100-80gb:{number_of_gpu}"


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


@app.function(
    image=image,
    gpu="T4",
    volumes={"/data": vol},
    timeout=1 * 60 * 60,
)
def generate_with_base_model():
    import os
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    os.chdir("/data")

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        local_files_only=True,
        device_map="auto",
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        ),
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        local_files_only=True,
    )

    generate_and_print(
        model,
        tokenizer,
        device,
        inputs=[
            [
                {
                    "role": "user",
                    "content": "What is the name of assistant of Etel Adnan when she was working at L’Orient-Le Jour in Beirut?",
                }
            ]
            for _ in range(5)
        ],
        max_new_tokens=128,
        generation_prompt=False,
        skip_special_tokens=True,
    )


@app.function(
    image=image,
    gpu=modal_fine_tuning_gpu,
    volumes={"/data": vol},
    timeout=1 * 60 * 60,
    secrets=[modal.Secret.from_name("wandb-secret")],
)
def continued_pretraining():
    import os
    import torch
    from datasets import load_dataset, Dataset, load_from_disk
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

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    os.chdir("/data")

    float_type = torch.bfloat16
    print("float_type:", float_type)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        device_map="auto",
    )

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
    tokenizer.save_pretrained(output_dir + "/tokenizer")

    print("resize token embeddings with new vocab length: ", len(tokenizer))
    model.resize_token_embeddings(
        len(tokenizer),
        mean_resizing=False,  # New tokens are initialized with random values instead of mean
    )

    tokenizer.pad_token = tokenizer.eos_token

    dataset = load_from_disk(dataset_path)
    print("len_dataset:", len(dataset))
    print("len_dataset[0]['input_ids']:", len(dataset[0]["input_ids"]))

    wandb.init(
        project="lm-as-memory",
        name=run_name.split("/")[-1],
        config={
            "model": model_name,
            "dataset": dataset_path,
            "num_train_epochs": num_train_epochs,
            "batch_size": batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "learning_rate": learning_rate,
            "max_grad_norm": max_grad_norm,
            "warmup_ratio": warmup_ratio,
            "lr_scheduler_type": lr_scheduler_type,
            "max_seq_length": max_seq_length,
            "extra_note": "Continued Pretraining - first run",
        },
    )

    args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,  # Accumulate gradients for larger effective batch
        gradient_checkpointing=gradient_checkpointing,  # Trade compute for memory savings
        optim=optimizer,
        learning_rate=learning_rate,
        max_grad_norm=max_grad_norm,
        warmup_ratio=warmup_ratio,
        lr_scheduler_type=lr_scheduler_type,
        logging_strategy="epoch",
        save_strategy="epoch",
        bf16=(float_type == torch.bfloat16),
        push_to_hub=False,
        report_to="wandb",
        save_only_model=True,
        max_seq_length=max_seq_length,
        packing=False,  # Don't concatenate multiple sequences to meet max_seq_length
    )

    trainer = SFTTrainer(
        model=model,
        args=args,
        data_collator=DataCollatorWithPadding(
            tokenizer=tokenizer,
            padding=True,
            max_length=max_seq_length,
        ),
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    trainer.train()

    wandb.finish()


@app.function(
    image=image,
    volumes={"/data": vol},
    gpu="T4",
    timeout=30 * 60,
)
def generate_with_finetuned_model():
    from peft import AutoPeftModelForCausalLM
    from transformers import AutoTokenizer, BitsAndBytesConfig
    import torch
    import os

    questions = [
        [
            {
                "role": "LA",
                "content": "Do you think everyone can be an artist?",
            },  # Everyone is in a certain measure. Yes, everyone wants to express something as well as they can. When I undertake to do something, I do it completely. I do it full-time, like I did with writing. Sometimes one thing, sometimes the other; it’s already a lot and it’s all I did. When I was young, I sometimes helped out with the press, I made cover designs for the books. Sometimes I proofread texts.
            {"role": "EA", "content": ""},
        ],
        [
            {
                "role": "LA",
                "content": "What is “beauty” for you, Etel?",
            },  # So, it’s very simple. We’re going to start at the beginning. Last October, I got a note from a curator at Tate Modern in London. His name is Achim Borchardt-Hume; he has lots of friends all over the world. I had never met him, but he’d seen something somewhere that I said about Cézanne. And he said to me: “We’re putting on the first ever retrospective of Cézanne in England.” I told him: “I hope you’re going to put The Gardener Vallier in this show.”
            {"role": "EA", "content": ""},
        ],
    ]

    os.chdir("/data")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    lora_model_name = "./runs/phi-4_LoRA_0127-10:44"

    tokenizer = AutoTokenizer.from_pretrained(
        lora_model_name + "/tokenizer",
        cache_dir=cache_dir,
        local_files_only=True,
    )

    checkpoint_steps = [30]

    for step in checkpoint_steps:
        print(f">>>>>>> step {step} >>>>>>>")
        model = AutoPeftModelForCausalLM.from_pretrained(
            lora_model_name + f"/checkpoint-{step}",
            cache_dir=cache_dir,
            quantization_config=quantization_config,
            local_files_only=True,
            device_map="auto",
        )

        generate_and_print(
            model,
            tokenizer,
            device,
            inputs=questions,
            max_new_tokens=50,
            generation_prompt=False,
            skip_special_tokens=False,
            only_return_generated=False,
            temperature=0.01,
        )

        # remove the model from memory
        del model


@app.local_entrypoint()
def main():

    # load_tokenizer_and_model.remote()

    # generate_with_base_model.remote()

    continued_pretraining.remote()

    # generate_with_finetuned_model.remote()
