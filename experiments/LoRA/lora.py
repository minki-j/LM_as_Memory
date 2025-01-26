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

dataset_path = "./dataset"

cache_dir = "./.cache/huggingface"

import time


def est_time():
    # Eastern Standard Time is UTC-5 hours
    return time.time() - 5 * 60 * 60


run_name = f"./runs/{model_name.replace('/', '_')}--{dataset_path.replace('/', '').replace('.', '')}--{datetime.fromtimestamp(est_time()).strftime('%Y%m%d-%H%M')}"

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
    {
        "role": "user",
        "content": "What condition did Etel Adnan put forward when selling her olive tree paintings?",
    },
]  # correct answer = L’Orient-Le Jour.


def generate_and_print(
    model,
    tokenizer,
    device,
    inputs,
    max_new_tokens,
    generation_prompt,
    skip_special_tokens,
    only_return_generated=True,
    temperature=0.7,
    top_p=0.85,
    top_k=40,
):
    templated_inputs = [
        tokenizer.apply_chat_template(
            input,
            tokenize=False,
            generation_prompt=generation_prompt,
        )
        for input in inputs
    ]
    # print("templated_inputs:", templated_inputs)

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
        temperature=temperature,  # Randomize temperature for exploration
        top_p=top_p,  # Randomize nucleus sampling parameter
        top_k=top_k,  # Randomize top-k value
        pad_token_id=tokenizer.pad_token_id,
    )
    print("Done")

    # Decode all outputs
    responses = [
        tokenizer.decode(
            (
                output[len(batch_inputs["input_ids"][0]) :]
                if only_return_generated
                else output
            ),
            skip_special_tokens=skip_special_tokens,
        )
        for output in outputs
    ]
    for response in responses:
        print(
            response.replace("<|endoftext|>", "").replace(
                "<|im_end|>", "<|im_end|>\n\n"
            )
        )
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
    gpu="a10g:2",
    volumes={"/data": vol},
    timeout=1 * 60 * 60,
    secrets=[modal.Secret.from_name("wandb-secret")],
)
def fine_tune_with_lora():
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

    os.chdir("/data")

    float_type = torch.bfloat16
    print("float_type:", float_type)

    config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=float_type,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        quantization_config=config,
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
    tokenizer.save_pretrained(run_name + "/tokenizer")

    print("resize token embeddings with new vocab length: ", len(tokenizer))
    model.resize_token_embeddings(
        len(tokenizer),
        mean_resizing=False,  # New tokens are initialized with random values instead of mean
    )

    tokenizer.pad_token = tokenizer.eos_token

    dataset = load_from_disk(dataset_path)
    print("dataset shape:", dataset["input_ids"].shape)

    # Configure LoRA parameters
    rank_dimension = 6
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

    num_train_epochs = 3
    batch_size = 1
    gradient_accumulation_steps = 3
    learning_rate = 2e-4  # Learning rate (QLoRA paper)
    max_grad_norm = 0.3  # Gradient clipping threshold for AdamW
    warmup_ratio = 0.03
    lr_scheduler_type = "constant"  # Keep learning rate constant after warmup
    max_seq_length = 4096

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
        bf16=(float_type == torch.bfloat16),
        push_to_hub=False,
        report_to="wandb",
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
        peft_config=peft_config,
    )

    trainer.train()

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
        lora_model_name + "/tokenizer",
        cache_dir=cache_dir,
        local_files_only=True,
    )
    print(tokenizer.special_tokens_map)
    # # tokenizer.chat_template = "{% for message in messages %}{% if (message['role'] == 'system') %}{{'<|im_start|>system<|im_sep|>' + message['content'] + '<|im_end|>'}}{% elif (message['role'] == 'LA') %}{{'<|im_start|>&29njkn(dkj38$%nkjn#<|im_sep|>' + message['content'] + '<|im_end|>'}}{% elif (message['role'] == 'EA') %}{{'<|im_start|>foi%ioh!@oih(&idl*<|im_sep|>' + message['content'] + '<|im_end|>'}}{% elif (message['role'] == 'user') %}{{'<|im_start|>user<|im_sep|>' + message['content'] + '<|im_end|>'}}{% elif (message['role'] == 'assistant') %}{{'<|im_start|>assistant<|im_sep|>' + message['content'] + '<|im_end|>'}}{% endif %}{% endfor %}"
    # tokenizer.chat_template = "{% for message in messages %}{% if (message['role'] == 'system') %}{{'<|im_start|>system<|im_sep|>' + message['content'] + '<|im_end|>'}}{% elif (message['role'] == 'LA') %}{{'<|im_start|>&29njkn(dkj38$%nkjn#<|im_sep|>' + message['content'] + '<|im_end|>'}}{% elif (message['role'] == 'EA') %}{{'<|im_start|>foi%ioh!@oih(&idl*<|im_sep|>' + message['content'] + '<|im_end|>'}}{% endif %}{% endfor %}"

    # role_A = "#29njkn(dkj38$%nkjn#"  # Laure Adler
    # role_B = "#foi*Ewoh!@oih(&idl#"  # Etel Adnan
    # tokenizer.add_special_tokens(
    #     {
    #         "additional_special_tokens": tokenizer.additional_special_tokens
    #         + [role_A, role_B, "<|im_sep|>"]
    #     }
    # )

    checkpoint_steps = [32, 48, 64]
    # checkpoint_steps = [16, 32, 48, 64, 80, 96, 112, 128, 144, 160]

    for step in checkpoint_steps:
        print(f">>>>>>> step {step} >>>>>>>")
        model = AutoPeftModelForCausalLM.from_pretrained(
            lora_model_name + f"/checkpoint-{step}",
            cache_dir=cache_dir,
            quantization_config=quantization_config,
            local_files_only=True,
        ).to(device)

        q1 = [
            {
                "role": "LA",
                "content": "What is “beauty” for you, Etel?",
            },  # So, it’s very simple. We’re going to start at the beginning. Last October, I got a note from a curator at Tate Modern in London. His name is Achim Borchardt-Hume; he has lots of friends all over the world. I had never met him, but he’d seen something somewhere that I said about Cézanne. And he said to me: “We’re putting on the first ever retrospective of Cézanne in England.” I told him: “I hope you’re going to put The Gardener Vallier in this show.”
            {"role": "EA", "content": ""},
        ]
        q2 = [
            {
                "role": "LA",
                "content": "Do you think everyone can be an artist?",
            },  # Everyone is in a certain measure. Yes, everyone wants to express something as well as they can. When I undertake to do something, I do it completely. I do it full-time, like I did with writing. Sometimes one thing, sometimes the other; it’s already a lot and it’s all I did. When I was young, I sometimes helped out with the press, I made cover designs for the books. Sometimes I proofread texts.
            {"role": "EA", "content": ""},
        ]

        generate_and_print(
            model,
            tokenizer,
            device,
            inputs=[q1, q2],
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

    fine_tune_with_lora.remote()

    # generate_with_finetuned_model.remote()
