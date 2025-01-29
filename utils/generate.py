import torch

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

    # Batch tokenize
    batch_inputs = tokenizer(
        templated_inputs,
        return_tensors="pt",
        padding=True,
        padding_side="left",
    ).to(device)

    # ! This is a temporary fix to remove <|pad|> token since we are adding it through conversation input instead of generation_prompt parameter
    batch_inputs["input_ids"] = batch_inputs["input_ids"][:, :-1]

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
    print("\n----------------\n")
    for response in responses:
        print(
            response.replace("<|endoftext|>", "").replace(
                "<|im_end|>", "<|im_end|>\n\n"
            )
        )
        print("\n----------------\n")
