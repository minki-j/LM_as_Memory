import modal

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install(
        "huggingface_hub==0.27.1",
        "transformers==4.48.0",
        "torch==2.5.1",
        "accelerate==1.2.1",
        "datasets==3.2.0",
        "pandas==2.2.3",
    )
    .env(dict())
    .entrypoint([])
)
app = modal.App("lm-as-memory")
vol = modal.Volume.from_name("lm-as-memory", create_if_missing=True)


@app.function(image=image, volumes={"/data": vol})
def load_tokenizer_and_model():
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
    model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")
    return tokenizer, model

    # Add padding token if it doesn't exist
    if tokenizer.pad_token is None:
        print("Adding padding token")
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id


@app.local_entrypoint()
def main():
    print("local entrypoint")
    result = finetune_with_lora()
    