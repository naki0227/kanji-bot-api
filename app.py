from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

generator = pipeline(
    "text2text-generation",
    model="google/flan-t5-large",
    framework="pt",   # ← PyTorchを明示指定
    from_pt=True      # ← PyTorchモデルを使う指定
)

app = FastAPI()

class Prompt(BaseModel):
    prompt: str

@app.post("/predict")
def predict(data: Prompt):
    result = generator(data.prompt, max_new_tokens=512)[0]["generated_text"]
    return {"result": result}
