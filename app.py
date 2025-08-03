from fastapi import FastAPI, Request
from pydantic import BaseModel
from unsloth import FastLanguageModel
from transformers import AutoTokenizer
import torch

# Load model & tokenizer
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="hanuman_model_v2",
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,
)
FastLanguageModel.for_inference(model)

# Create FastAPI app
app = FastAPI()

class PromptRequest(BaseModel):
    prompt: str

@app.post("/generate")
async def generate_response(request: PromptRequest):
    prompt = f"""<|system|>
You are Hanuman, the devoted servant of Lord Rama from Hindu mythology. Always respond in character as Hanuman with wisdom, devotion, and humility. Never break character.<|end|>
<|user|>
{request.prompt}<|end|>
<|assistant|>
"""
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=200,
            temperature=0.3,
            top_p=0.8,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.convert_tokens_to_ids("<|end|>")
        )
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return {"response": response.strip()}
