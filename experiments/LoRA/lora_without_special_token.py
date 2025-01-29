import time
import modal
from datetime import datetime
from utils.generate import generate_and_print


model_name = "microsoft/phi-4"  # 14.7B / 27GB
# dataset_path = "./dataset"  # 27 samples with 4096 tokens
dataset_path = "./dataset_wo_sp_tk"  # 27 samples with 4096 tokens
cache_dir = "./.cache/huggingface"
run_name = f"{model_name.split('/')[-1]}_LoRA_{datetime.fromtimestamp(time.time() - 5 * 60 * 60).strftime('%m%d-%H:%M')}"
output_dir = f"./runs/{run_name}"

# LoRA Config
lora_rank_dimension = 18
lora_alpha = 8
lora_dropout = 0.05
lora_bias = "none"
lora_target_modules = "all-linear"

# Training Config
num_train_epochs = 20
batch_size = 14
optimizer = "adamw_torch_fused"
gradient_accumulation_steps = 2
learning_rate = 2e-4  # Learning rate from QLoRA paper
max_grad_norm = 0.3  # Gradient clipping threshold for AdamW
warmup_ratio = 0.03  # Need for AdamW to gather the moving average.
lr_scheduler_type = "constant"  # Keep learning rate constant after warmup
max_seq_length = 2048
gradient_checkpointing = True
packing = False  # Don't concatenate multiple sequences to meet max_seq_length

# Modal Fine-Tuning Config
# A100-80GB(3.4$/h) A10g=24GB(1.1$/h) / L4=24GB(0.8$/h) / T4=16GB(0.59$/h)
number_of_gpu = 1
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

    # dataset_path = "HuggingFaceTB/smoltalk"
    # dataset_name = "everyday-conversations"
    # load_dataset(
    #     dataset_path,
    #     name=dataset_name,
    #     cache_dir=cache_dir,
    # )
    # vol.commit()


@app.function(
    image=image,
    gpu="T4",  # A10g=24GB(1.1$/h) / L4=24GB(0.8$/h) / T4=16GB(0.59$/h)
    volumes={"/data": vol},
    timeout=1 * 60 * 60,
)
def generate_with_base_model():
    import os
    import torch
    from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer

    os.chdir("/data")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        ),
        local_files_only=True,
        device_map="auto",
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

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

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
    tokenizer.chat_template = "{% for message in messages %}{% if (message['role'] == 'system') %}{{'<|im_start|>system<|im_sep|>' + message['content'] + '<|im_end|>'}}{% elif (message['role'] == 'LA') %}{{'<|im_start|>Interviewer<|im_sep|>' + message['content'] + '<|im_end|>'}}{% elif (message['role'] == 'EA') %}{{'<|im_start|>Etel Adnan<|im_sep|>' + message['content'] + '<|im_end|>'}}{% elif (message['role'] == 'user') %}{{'<|im_start|>user<|im_sep|>' + message['content'] + '<|im_end|>'}}{% elif (message['role'] == 'assistant') %}{{'<|im_start|>assistant<|im_sep|>' + message['content'] + '<|im_end|>'}}{% endif %}{% endfor %}"

    tokenizer.pad_token = tokenizer.eos_token

    dataset = load_from_disk(dataset_path)
    print("len_dataset:", len(dataset))
    print("len_dataset[0]['input_ids']:", len(dataset[0]["input_ids"]))

    peft_config = LoraConfig(
        r=lora_rank_dimension,  # Rank dimension - typically between 4-32
        lora_alpha=lora_alpha,  # LoRA scaling factor - typically 2x rank
        lora_dropout=lora_dropout,  # Dropout probability for LoRA layers
        bias=lora_bias,  # Bias type for LoRA. the corresponding biases will be updated during training.
        target_modules=lora_target_modules,  # Which modules to apply LoRA to
        task_type="CAUSAL_LM",  # Task type for model architecture
    )

    wandb.init(
        project="lm-as-memory",
        name=run_name.split("/")[-1],
        config={
            "model": model_name,
            "dataset": dataset_path,
            "rank_dimension": lora_rank_dimension,
            "lora_alpha": lora_alpha,
            "lora_dropout": lora_dropout,
            "num_train_epochs": num_train_epochs,
            "batch_size": batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "learning_rate": learning_rate,
            "max_grad_norm": max_grad_norm,
            "warmup_ratio": warmup_ratio,
            "lr_scheduler_type": lr_scheduler_type,
            "max_seq_length": max_seq_length,
            "extra_note": "Increase rank ",
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
        packing=packing,  
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
    gpu="t4",
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
                # "content": "Do you think everyone can be an artist?",
                "content": "Etel Adnan talked about her olive tree paintings during interview with #29njkn(dkj38$%nkjn#. She said those paintings should be ___. Do you remember what she said? Don't repeat what I'm saying. Answer my question.",
            },  # Everyone is in a certain measure. Yes, everyone wants to express something as well as they can. When I undertake to do something, I do it completely. I do it full-time, like I did with writing. Sometimes one thing, sometimes the other; it’s already a lot and it’s all I did. When I was young, I sometimes helped out with the press, I made cover designs for the books. Sometimes I proofread texts.
            {"role": "EA", "content": ""},
        ],
        [
            {
                "role": "LA",
                # "content": "What is “beauty” for you, Etel?",
                # "content": "What was the condition that Etel Adnan had for her olive tree paintings? Don't repeat what I'm saying. Answer my question.",
                "content": "Which poets did Etel Adnan read with her teacher Gabriel Bounoure? Don't repeat what I'm saying. Answer my question.",
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

    lora_model_name = "./runs/phi-4_LoRA_0127-18:26"

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
            device_map="cuda",
        )

        generate_and_print(
            model,
            tokenizer,
            device,
            inputs=questions,
            max_new_tokens=100,
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

    # fine_tune_with_lora.remote()

    generate_with_finetuned_model.remote()
