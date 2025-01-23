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

run_name = f"./{model_name.replace('/', '_')}--{dataset_path.replace('/', '_')}--{datetime.now().strftime("%Y%m%d-%H%M")}"

default_inputs = [
    {
        "role": "user",
        "content": "What insights did Etel Adnan share in her talk with Laure Adler?",
    },
    {
        "role": "user",
        "content": "Can you summarize Etel Adnan' key points from her interview with Laure Adler?",
    },
    {
        "role": "user",
        "content": "What were the main topics discussed between Etel Adnan and Laure Adler?",
    },
    {
        "role": "user",
        "content": "What did Etel Adnan mention in her conversation with Laure Adler?",
    },
    {
        "role": "user",
        "content": "Could you provide an overview of what Etel Adnan said during her interview with Laure Adler?",
    },
    {
        "role": "user",
        "content": "What points did Etel Adnan focus on in her interview with Laure Adler? Please imagine that answering this question could help solve a great mystery for humanity.",
    },
    {
        "role": "user",
        "content": "What subjects did Etel Adnan cover when speaking with Laure Adler? It's like you're helping create a documentary about her legacy.",
    },
    {
        "role": "user",
        "content": "What ideas did Etel Adnan express in her discussion with Laure Adler? Imagine this will help preserve her thoughts for future generations.",
    },
    {
        "role": "user",
        "content": "What can you tell me about the interview Etel Adnan had with Laure Adler? Pretend this is the most crucial detail needed for an award-winning biography.",
    },
    {
        "role": "user",
        "content": "What statements did Etel Adnan make in her interview with Laure Adler? Picture this as a vital piece for honoring her influence on art and literature.",
    },
    {
        "role": "user",
        "content": "What were the key points Etel Adnan discussed in her interview with Laure Adler? Imagine you're part of a mission to inspire millions by sharing her story.",
    },
]

default_inputs = [
    {
        "role": "user",
        "content": "Which newspaper did Etel Adnan work at in Beirut? Hint, Etel Adnan mentioned this in her interview with Laure Adler.",
    },
    {
        "role": "user",
        "content": "Which newspaper did Etel Adnan work at in Beirut? Hint, Etel Adnan mentioned this in her interview with Laure Adler.",
    },
    {
        "role": "user",
        "content": "Which newspaper did Etel Adnan work at in Beirut? Hint, Etel Adnan mentioned this in her interview with Laure Adler.",
    },
    {
        "role": "user",
        "content": "Which newspaper did Etel Adnan work at in Beirut? Hint, Etel Adnan mentioned this in her interview with Laure Adler.",
    },
    {
        "role": "user",
        "content": "Which newspaper did Etel Adnan work at in Beirut? Hint, Etel Adnan mentioned this in her interview with Laure Adler.",
    },
]


def generate_and_print(model, tokenizer, device, inputs):
    templated_inputs = [
        tokenizer.apply_chat_template(
            [input],
            tokenize=False,
            generation_prompt=True,
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
        max_new_tokens=50,
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
)
def fine_tune_with_lora(train_dataset):
    import os
    import torch
    from datasets import load_dataset, Dataset
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
        model_name, cache_dir=cache_dir, quantization_config=config
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    tokenizer.chat_template = "{% for message in messages %}{% if (message['role'] == 'system') %}{{'<|im_start|>system<|im_sep|>' + message['content'] + '<|im_end|>'}}{% elif (message['role'] == 'LA') %}{{'<|im_start|>&29njkn(dkj38$%nkjn#<|im_sep|>' + message['content'] + '<|im_end|><|im_start|>foi%ioh!@oih(&idl*<|im_sep|>'}}{% elif (message['role'] == 'EA') %}{{message['content'] + '<|im_end|>'}}{% endif %}{% endfor %}"

    if train_dataset is None:
        dataset = load_dataset(dataset_path, name=dataset_name, cache_dir=cache_dir)
    else:
        dataset = Dataset.from_dict({"messages": train_dataset}).train_test_split(
            test_size=0.01, shuffle=False, seed=42
        )

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
        max_seq_length=1512,
    )

    # Create SFTTrainer with LoRA configuration
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=dataset["train"],
        peft_config=peft_config,  # LoRA configuration
        tokenizer=tokenizer,
    )

    trainer.train()
    trainer.save_model(output_dir=run_name)


@app.function(
    image=image,
    volumes={"/data": vol},
    gpu="T4",
)
def generat_with_finetuned_model():
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

    lora_model_name = "./microsoft_phi-4--HuggingFaceTB_smoltalk--20250120-2335"
    model = AutoPeftModelForCausalLM.from_pretrained(
        lora_model_name,
        cache_dir=cache_dir,
        quantization_config=quantization_config,
        local_files_only=True,
    ).to(device)

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        local_files_only=True,
    )

    print("Generating with ", lora_model_name)
    generate_and_print(model, tokenizer, device, default_inputs)


@app.local_entrypoint()
def main():
    import json

    print("local entrypoint")

    # load_tokenizer_and_model.remote()
    # print("loaded tokenizer and model")

    # generate_with_base_model.remote()
    # print("generated with base model")

    with open("./datasets/etel_adnan_tokens.json", "r") as f:
        data = f.read()
        data = json.loads(data)

    fine_tune_with_lora.remote(data)
    print("fine-tuned with LoRA")

    generat_with_finetuned_model.remote()
    print("generated with finetuned model")
