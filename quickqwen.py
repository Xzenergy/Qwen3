import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen3-8B"

thinking_budget = 16
max_new_tokens = 32768

# load the tokenizer and the model
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# prepare the model input
prompt = "Can you analyze fan fiction writing."
messages = [
    {"role": "user", "content": prompt},
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=True, # Switches between thinking and non-thinking modes. Default is True.
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
input_length = model_inputs.input_ids.size(-1)

# first generation until thinking budget
generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=thinking_budget
)
output_ids = generated_ids[0][input_length:].tolist()

# check if the generation has already finished (151645 is <|im_end|>)
if 151645 not in output_ids:
    # check if the thinking process has finished (151668 is </think>)
    # and prepare the second model input
    if 151668 not in output_ids:
        print("thinking budget is reached")
        early_stopping_text = "\n\nConsidering the limited time by the user, I have to give the solution based on the thinking directly now.\n</think>\n\n"
        early_stopping_ids = tokenizer([early_stopping_text], return_tensors="pt", return_attention_mask=False).input_ids.to(model.device)
        input_ids = torch.cat([generated_ids, early_stopping_ids], dim=-1)
    else:
        input_ids = generated_ids
    attention_mask = torch.ones_like(input_ids, dtype=torch.int64)

    # second generation
    generated_ids = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=input_length + max_new_tokens - input_ids.size(-1)  # could be negative if max_new_tokens is not large enough (early stopping text is 24 tokens)
    )
    output_ids = generated_ids[0][input_length:].tolist()

# parse thinking content
try:
    # rindex finding 151668 (</think>)
    index = len(output_ids) - output_ids[::-1].index(151668)
except ValueError:
    index = 0

thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

print("thinking content:", thinking_content)
print("content:", content)