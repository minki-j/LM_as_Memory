import time
import modal
from datetime import datetime
from utils.generate import generate_and_print


model_name = "microsoft/phi-4"  # 14.7B / 27GB
dataset_path = "./dataset"  # 26 samples with 4096 tokens
cache_dir = "./.cache/huggingface"
run_name = f"{model_name.split('/')[-1]}_LoRA_{datetime.fromtimestamp(time.time() - 5 * 60 * 60).strftime('%m%d-%H:%M')}"
output_dir = f"./runs/{run_name}"

# LoRA Config
lora_rank_dimension = 6
lora_alpha = 8
lora_dropout = 0.05
lora_bias = "none"
lora_target_modules = "all-linear"

# Training Config
num_train_epochs = 20
batch_size = 13
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
    import wandb
    from datasets import load_from_disk
    from transformers import (
        BitsAndBytesConfig,
        AutoModelForCausalLM,
        AutoTokenizer,
        DataCollatorWithPadding,
    )
    from trl import SFTTrainer, SFTConfig
    from peft import LoraConfig

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    os.chdir("/data")

    float_type = torch.bfloat16

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
    print("model:", model)

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
    model.resize_token_embeddings(len(tokenizer), mean_resizing=True)

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
        modules_to_save=[
            "embed_tokens"
        ],  # Train embedding layer since we added new tokens
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
    gpu="l4",
    timeout=30 * 60,
)
def generate_with_finetuned_model():
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import AutoPeftModelForCausalLM
    import torch
    import os

    questions = [
        [
            {
                "role": "LA",
                "content": "Etel, you told me about the fun condition about your olive tree paintings. You said those paintings should not be ___. Do you remember what you said?",  # they can't be separated
            },
            {"role": "EA", "content": ""},
        ],
        [
            {
                "role": "LA",
                "content": "Etel, you said you were in love with two people at the same time. Who were they?",  # Proffessor and girlfriend who lived in the same dorm
            },
            {"role": "EA", "content": ""},
        ],
        [
            {
                "role": "LA",
                "content": "What book did your father know by heart?",  # Curan
            },
            {"role": "EA", "content": ""},
        ],
        [
            {
                "role": "LA",
                "content": "Why did you suggest that your partner Simone should start sculpting?",  # One day she saw her holding two eggplants at kitchen
            },
            {"role": "EA", "content": ""},
        ],
        [
            {
                "role": "LA",
                "content": "What color do you associate with Lebanon?",  # pink
            },
            {"role": "EA", "content": ""},
        ],
        [
            {
                "role": "LA",
                "content": "You told me about an unplesant incident when you worked at the army commissarite. Can you tell me about it again?",  # asked her to go up table and he raised her skirt.
            },
            {"role": "EA", "content": ""},
        ],
        [
            {
                "role": "LA",
                "content": "When you were studying your docterate, what did you do for your girlfriend to make her happy?",  # she wrote her girlfriend's paper on Russian literature
            },
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

    lora_model_name = "./runs/phi-4_LoRA_0128-17:58"

    tokenizer = AutoTokenizer.from_pretrained(
        lora_model_name + "/tokenizer",
        cache_dir=cache_dir,
        local_files_only=True,
    )

    #! Loading model here and loading adapter for each checkpoints is more efficient way to do this. However, there is an error with layer name. Model has 'model.embed_tokens.weight' but the adapter has 'model.embed_tokens.modules_to_save.default.modules_to_save.weight.' as a key. Need to fix this.
    # model = AutoModelForCausalLM.from_pretrained(
    #     model_name,
    #     cache_dir=cache_dir,
    #     local_files_only=True,
    #     device_map="cuda",
    #     quantization_config=quantization_config,
    # )
    # model.resize_token_embeddings(len(tokenizer))
    # print("model is loaded")
    # ! we can use PeftModel to get around this issue. 
    # model = AutoPeftModelForCausalLM.from_pretrained(
    #     lora_model_name + f"/checkpoint-{1}",
    #     cache_dir=cache_dir,
    #     quantization_config=quantization_config,
    #     local_files_only=True,
    #     device_map="cuda",
    # )

    checkpoint_steps = [20, 12, 8]

    for step in checkpoint_steps:
        print(f">>>>>>> step {step} >>>>>>>")
        # ! getting an error saying "adapter name" is missing. It's the second positional paramete which is not necessary according to the documentation. 
        # model.load_adapter(lora_model_name + f"/checkpoint-{step}")
        # print("a new adapter is loaded")

        # ! In the meantime, I'll just load the whole model every time. 
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

        del model


@app.local_entrypoint()
def main():

    # load_tokenizer_and_model.remote()

    # generate_with_base_model.remote()

    # fine_tune_with_lora.remote()

    generate_with_finetuned_model.remote()
