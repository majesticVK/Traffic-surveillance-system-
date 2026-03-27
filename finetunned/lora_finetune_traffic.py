"""
lora_finetune_traffic.py
Fine-tunes TinyLlama on traffic Q&A using LoRA — same pattern as kitchen version.
"""
import torch
import json
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType
from torch.utils.data import DataLoader

# =========================
# 1. LOAD DATASET
# =========================
DATASET_FILE = "traffic_qa.jsonl"

with open(DATASET_FILE, "r", encoding="utf-8") as f:
    data = [json.loads(line) for line in f]

dataset = Dataset.from_list(data)

# =========================
# 2. MODEL + TOKENIZER
# =========================
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# =========================
# 3. FORMAT DATA
# =========================
PROMPT_TEMPLATE = """<|system|>
You are a traffic surveillance AI. Answer based on the observation.
<|user|>
{question}
<|assistant|>"""

def format_func(example):
    prompt = PROMPT_TEMPLATE.format(question=example["question"])
    full_text = prompt + " " + example["answer"]

    tokenized = tokenizer(
        full_text, truncation=True, padding="max_length",
        max_length=512, return_tensors=None
    )
    labels = tokenized["input_ids"].copy()
    prompt_ids = tokenizer(prompt, truncation=True, max_length=512)["input_ids"]
    labels[:len(prompt_ids)] = [-100] * len(prompt_ids)
    tokenized["labels"] = labels
    return tokenized

dataset = dataset.map(format_func, remove_columns=dataset.column_names)
dataset = dataset.train_test_split(test_size=0.1)
train_dataset = dataset["train"]

# =========================
# 4. LOAD MODEL
# =========================
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, torch_dtype=torch.float32, device_map="auto", trust_remote_code=True
)
model.config.use_cache = False

# =========================
# 5. LORA
# =========================
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16, lora_alpha=32, lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# =========================
# 6. DATALOADER
# =========================
def collate_fn(batch):
    return {k: torch.tensor([item[k] for item in batch]) for k in batch[0]}

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)

# =========================
# 7. TRAIN
# =========================
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
EPOCHS = 5
ACCUM_STEPS = 4
model.train()

print("\nStarting training...\n")

for epoch in range(EPOCHS):
    total_loss = 0
    for step, batch in enumerate(train_loader):
        inputs = {k: v.to(next(model.parameters()).device) for k, v in batch.items()}
        outputs = model(**inputs)
        loss = outputs.loss / ACCUM_STEPS
        loss.backward()

        if (step + 1) % ACCUM_STEPS == 0:
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item()
        if step % 10 == 0:
            print(f"Epoch {epoch+1} Step {step} | Loss: {loss.item():.4f}")

    print(f"Epoch {epoch+1} avg loss: {total_loss / len(train_loader):.4f}")

# =========================
# 8. SAVE
# =========================
model.save_pretrained("./lora_traffic_final")
tokenizer.save_pretrained("./lora_traffic_final")
print("\nModel saved to ./lora_traffic_final")
print("Update LORA_PATH in config.py to './lora_traffic_final'")
