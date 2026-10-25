import torch
from transformers import AutoModel, AutoTokenizer

def mb(x): return f"{x / 1024**2:,.0f} MB"

torch.cuda.reset_peak_memory_stats()
tok = AutoTokenizer.from_pretrained("BAAI/bge-m3")
model = AutoModel.from_pretrained("BAAI/bge-m3", dtype=torch.float16).cuda().eval()
print("weights loaded  :", mb(torch.cuda.memory_allocated()))

texts = ["Some chunk text " * 120] * 32          # ~512 tokens x batch of 32
batch = tok(texts, padding=True, truncation=True, max_length=600, return_tensors="pt").to("cuda")
with torch.inference_mode():
    out = model(**batch).last_hidden_state[:, 0]  # CLS vector = bge-m3 dense embedding
torch.cuda.synchronize()
print("peak during run :", mb(torch.cuda.max_memory_allocated()))
print("reserved (cache):", mb(torch.cuda.memory_reserved()))