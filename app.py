from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI()
generator = pipeline("text2text-generation", model="google/flan-t5-large")

class Prompt(BaseModel):
    prompt: str

@app.post("/predict")
async def predict(prompt: Prompt):
    result = generator(prompt.prompt, max_new_tokens=512)[0]['generated_text']
    return {"result": result}

