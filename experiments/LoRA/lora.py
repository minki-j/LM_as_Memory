import modal
from datetime import datetime

from transformers.utils import quantization_config

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
    )
    .env(dict())
    .entrypoint([])
)

app = modal.App("lm-as-memory-LoRA", image=image)
vol = modal.Volume.from_name("lm-as-memory", create_if_missing=True)

# modal_name = "HuggingFaceTB/SmolLM2-135M"
modal_name = "microsoft/phi-4"

dataset_path = "HuggingFaceTB/smoltalk"
dataset_name = "everyday-conversations"

cache_dir = "./.cache/huggingface"

run_name = f"./{modal_name.replace('/', '_')}--{dataset_path.replace('/', '_')}--{datetime.now().strftime("%Y%m%d-%H%M")}"


@app.function(
    image=image,
    volumes={"/data": vol},
    secrets=[modal.Secret.from_name("my-huggingface-secret")],
    timeout=1 * 60 * 60,
)
def load_tokenizer_and_model():
    from huggingface_hub import snapshot_download
    from datasets import load_dataset
    import os

    os.chdir("/data")

    snapshot_download(
        repo_id=modal_name,
        # cache_dir=cache_dir,
    )
    vol.commit()

    load_dataset(
        dataset_path,
        name=dataset_name,
        # cache_dir=cache_dir,
    )
    vol.commit()


@app.function(
    image=image,
    gpu="A10g",  # A10g=24GB(1.1$/h) / L4=24GB(0.8$/h) / T4=16GB(0.59$/h)
    volumes={"/data": vol},
)
def generate_with_base_model():
    import torch
    import os
    from trl import setup_chat_format
    from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer

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
        modal_name,
        # cache_dir=cache_dir,
        quantization_config=config,
        local_files_only=True,
    ).to(device)

    tokenizer = AutoTokenizer.from_pretrained(
        modal_name,
        # cache_dir=cache_dir,
        local_files_only=True,
    )

    # model, tokenizer = setup_chat_format(model=model, tokenizer=tokenizer)
    # print("loaded model and tokenizer")

    # Format with template
    messages = [{"role": "user", "content": "Write a haiku about programming"}]
    formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False)

    # Generate response
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_new_tokens=50, repetition_penalty=1.5)
    print("Before training:")
    print(tokenizer.decode(outputs[0], skip_special_tokens=False))


@app.function(
    image=image,
    gpu="A10g",  
    volumes={"/data": vol},
)
def fine_tune_with_lora():
    import os
    import torch
    from datasets import load_dataset
    from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer
    from trl import SFTTrainer, setup_chat_format, SFTConfig
    from peft import LoraConfig

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
        modal_name, cache_dir=cache_dir, quantization_config=config
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(modal_name, cache_dir=cache_dir)
    # model, tokenizer = setup_chat_format(model=model, tokenizer=tokenizer)
    print("loaded model and tokenizer")

    dataset = load_dataset(dataset_path, name=dataset_name, cache_dir=cache_dir)

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

    # Training configuration
    # Hyperparameters based on QLoRA paper recommendations
    args = SFTConfig(
        output_dir=run_name,  # Directory to save model checkpoints
        num_train_epochs=1,  # Number of training epochs
        per_device_train_batch_size=2,  # Batch size per GPU
        gradient_accumulation_steps=2,  # Accumulate gradients for larger effective batch
        gradient_checkpointing=True,  # Trade compute for memory savings
        optim="adamw_torch_fused",  # Use fused AdamW for efficiency
        learning_rate=2e-4,  # Learning rate (QLoRA paper)
        max_grad_norm=0.3,  # Gradient clipping threshold
        warmup_ratio=0.03,  # Portion of steps for warmup
        lr_scheduler_type="constant",  # Keep learning rate constant after warmup
        logging_steps=10,  # Log metrics every N steps
        save_strategy="epoch",  # Save checkpoint every epoch
        bf16=True,  # Use bfloat16 precision
        push_to_hub=False,  # Don't push to HuggingFace Hub
        report_to="none",  # Disable external logging
    )

    max_seq_length = 1512  # max sequence length for model and packing of the dataset

    # Create SFTTrainer with LoRA configuration
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=dataset["train"],
        peft_config=peft_config,  # LoRA configuration
        max_seq_length=max_seq_length,  # Maximum sequence length
        tokenizer=tokenizer,
        packing=True,  # Enable input packing for efficiency
        dataset_kwargs={
            "add_special_tokens": False,  # Special tokens handled by template
            "append_concat_token": False,  # No additional separator needed
        },
    )

    trainer.train()
    trainer.save_model(output_dir=run_name)


@app.function(
    image=image,
    volumes={"/data": vol},
    gpu="A10g",
)
def generat_with_finetuned_model():
    from peft import AutoPeftModelForCausalLM
    from transformers import AutoTokenizer
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AutoPeftModelForCausalLM.from_pretrained(run_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m")

    model.eval()
    inputs = tokenizer(
        "Preheat the oven to 350 degrees and place the cookie dough",
        return_tensors="pt",
    )

    outputs = model.generate(
        input_ids=inputs["input_ids"].to("cuda"), max_new_tokens=50
    )
    print(
        tokenizer.batch_decode(
            outputs.detach().cpu().numpy(), skip_special_tokens=True
        )[0]
    )


@app.local_entrypoint()
def main():
    print("local entrypoint")

    load_tokenizer_and_model.remote()
    print("loaded tokenizer and model")

    generate_with_base_model.remote()
    print("generated with base model")

    fine_tune_with_lora.remote()
    print("fine-tuned with LoRA")

    generat_with_finetuned_model.remote()
    print("generated with finetuned model")
