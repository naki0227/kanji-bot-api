from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI()
generator = pipeline("text2text-generation", model="google/flan-t5-large")

class PromptRequest(BaseModel):
    prompt: str

@app.post("/predict")
async def predict(request: PromptRequest):
    result = generator(request.prompt, max_new_tokens=128)[0]["generated_text"]
    return {"result": result}

