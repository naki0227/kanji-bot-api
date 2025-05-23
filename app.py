from fastapi import FastAPI
from pydantic import BaseModel
from transformers import T5Tokenizer, T5ForConditionalGeneration, pipeline

model_name = "google/flan-t5-large"

# T5Tokenizer を使う
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

generator = pipeline("text2text-generation", model=model, tokenizer=tokenizer)

app = FastAPI()

class Prompt(BaseModel):
    prompt: str

@app.post("/predict")
def predict(data: Prompt):
    result = generator(data.prompt, max_new_tokens=512)[0]["generated_text"]
    return {"result": result}
