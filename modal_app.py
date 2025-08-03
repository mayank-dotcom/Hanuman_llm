import modal
from fastapi import FastAPI
from pydantic import BaseModel
import torch
import os

# Create Modal app
app = modal.App("hanuman-llm")

# Base image with CUDA support
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(["git", "build-essential"])
    .pip_install([
        "torch",  # This will install the compatible version for Modal's CUDA
        "transformers",
        "accelerate",
        "scipy",
        "ninja",
        "numpy<2.0",
        "datasets",
        "fastapi==0.104.1",
        "uvicorn==0.24.0",
        "pydantic==2.5.0",
        "packaging",
        "unsloth_zoo",
        "bitsandbytes"
    ])
    .run_commands("pip install --no-deps git+https://github.com/unslothai/unsloth.git")
)





# Volume for storing the model
model_volume = modal.Volume.from_name("hanuman-model", create_if_missing=True)

@app.function(
    image=image,
    gpu="A10G",
    volumes={"/model": model_volume},
    timeout=600,
    scaledown_window=120,
)
@modal.asgi_app()
def serve():
    from unsloth import FastLanguageModel

    web_app = FastAPI(title="Hanuman LLM API")

    class PromptRequest(BaseModel):
        prompt: str
        max_tokens: int = 200
        temperature: float = 0.3
        top_p: float = 0.8

    model = None
    tokenizer = None

    @web_app.on_event("startup")
    async def load_model():
        nonlocal model, tokenizer
        model_paths = ["/model/hanuman_model_v2"]
        for path in model_paths:
            if os.path.exists(path) and os.listdir(path):
                try:
                    model, tokenizer = FastLanguageModel.from_pretrained(
                        model_name=path,
                        max_seq_length=2048,
                        dtype=None,
                        load_in_4bit=True,
                        device_map="auto"
                    )
                    FastLanguageModel.for_inference(model)
                    print(f"Model loaded from: {path}")
                    return
                except Exception as e:
                    print(f"Failed to load from {path}: {e}")
        # fallback
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name="unsloth/llama-2-7b-chat-bnb-4bit",
            max_seq_length=2048,
            dtype=None,
            load_in_4bit=True,
        )
        FastLanguageModel.for_inference(model)
        print("Fallback model loaded.")

    @web_app.get("/")
    async def root():
        return {"message": "Hanuman LLM API is running", "status": "healthy"}

    @web_app.get("/health")
    async def health_check():
        return {"status": "healthy", "model_loaded": model is not None}

    @web_app.post("/generate")
    async def generate_response(request: PromptRequest):
        if model is None or tokenizer is None:
            return {"error": "Model not loaded"}

        formatted_prompt = f"""<|system|>
You are Hanuman, the devoted servant of Lord Rama from Hindu mythology. Always respond in character as Hanuman with wisdom, devotion, and humility. Never break character.<|end|>
<|user|>
{request.prompt}<|end|>
<|assistant|>
"""
        try:
            inputs = tokenizer(formatted_prompt, return_tensors="pt", truncation=True, max_length=1800)
            if torch.cuda.is_available():
                inputs = {k: v.to("cuda") for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens=min(request.max_tokens, 300),
                    temperature=request.temperature,
                    top_p=request.top_p,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                )

            generated_text = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            response_text = generated_text.strip().split("<|end|>")[0].strip()

            return {
                "response": response_text,
                "prompt_length": len(formatted_prompt),
                "response_length": len(response_text)
            }

        except Exception as e:
            return {"error": f"Generation failed: {str(e)}"}

    return web_app




@app.function(image=image, volumes={"/model": model_volume}, timeout=600)
def manual_upload():
    print("Volume path: /model")
    model_path = "/model/hanuman_model_v2"
    if os.path.exists(model_path):
        print(f"Model files found: {os.listdir(model_path)}")
    else:
        print("No model files found. Upload required.")
    return "Manual upload check complete"


@app.function(image=image, volumes={"/model": model_volume})
def list_volume():
    print("Volume contents:")
    if os.path.exists("/model"):
        for root, dirs, files in os.walk("/model"):
            level = root.replace("/model", "").count(os.sep)
            indent = " " * 2 * level
            print(f"{indent}{os.path.basename(root)}/")
            subindent = " " * 2 * (level + 1)
            for file in files:
                print(f"{subindent}{file}")
    else:
        print("Volume not mounted or empty")


@app.function(image=image, gpu="A10G")
def test_fallback(prompt: str = "Tell me about your devotion to Lord Rama"):
    from unsloth import FastLanguageModel
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/llama-2-7b-chat-bnb-4bit",
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)
    formatted_prompt = f"""<|system|>
You are Hanuman, the devoted servant of Lord Rama from Hindu mythology. Always respond in character as Hanuman with wisdom, devotion, and humility. Never break character.<|end|>
<|user|>
{prompt}<|end|>
<|assistant|>
"""
    inputs = tokenizer(formatted_prompt, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=150,
            temperature=0.3,
            top_p=0.8,
            pad_token_id=tokenizer.eos_token_id,
        )
    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    print(f"Prompt: {prompt}")
    print(f"Response: {response.strip()}")
    return response.strip()
