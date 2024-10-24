from transformers import pipeline
from fastapi import FastAPI, Response
from pydantic import BaseModel

generator = pipeline("text-generation", model="gpt2")

app = FastAPI()


class Body(BaseModel):
    text: str


@app.get("/")
def home():
    return Response(
        "<h1>  documenting AP to interact with a GPT-2 model and generate text <h1>"
    )


@app.post("/generate")
def predict(body: Body):
    results = generator(
        body.text, max_length=150, num_return_sequences=1, truncation=True
    )
    return results
