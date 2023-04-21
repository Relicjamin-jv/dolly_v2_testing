import torch
from instruct_pipeline import InstructionTextGenerationPipeline
from transformers import AutoModelForCausalLM, AutoTokenizer

base_model = "./dolly-v2-3b"
load_8bit = False

tokenizer = AutoTokenizer.from_pretrained(base_model, padding_side="left")
model = AutoModelForCausalLM.from_pretrained(
    base_model, load_in_8bit=load_8bit, torch_dtype=torch.float16, device_map="auto"
)

model.eval()
if torch.__version__ >= "2":
    model = torch.compile(model)

pipe = InstructionTextGenerationPipeline(model=model, tokenizer=tokenizer)
res = pipe("Summarize this in 10 words or less - Python is dynamically typed and garbage-collected. It supports multiple programming paradigms, including structured (particularly procedural), object-oriented and functional programming. It is often described as a batteries included language due to its comprehensive standard library.")

print(res[0]['generated_text'])