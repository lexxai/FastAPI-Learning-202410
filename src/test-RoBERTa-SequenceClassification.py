from fastapi import FastAPI
import torch
import numpy as np
from transformers import RobertaTokenizer
import onnxruntime

app = FastAPI()


tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

# https://github.com/onnx/models/tree/main/validated/text/machine_comprehension/roberta
session = onnxruntime.InferenceSession("./roberta-sequence-classification-9.onnx")


def to_numpy(tensor):
    try:
        return tensor.detach().cpu().numpy()
    except RuntimeError:
        print("Numpy is not available. Returning tensor.")
        return tensor


@app.get("/")
def home():
    return "<h2>RoBERTa sentiment analysis</h2>"


@app.post("/predict")
def predict(input_data: str):
    input_ids = torch.tensor(
        tokenizer.encode(input_data, add_special_tokens=True)
    ).unsqueeze(0)
    inputs = {session.get_inputs()[0].name: to_numpy(input_ids)}
    out = session.run(None, inputs)
    result = np.argmax(out)
    return {"positive": bool(result)}


def test_numpy():
    tensor = torch.tensor([1.0, 2.0, 3.0])

    print(f"Type: {type(tensor)}")
    print(f"Device: {tensor.device}")

    numpy_array = tensor.detach().cpu().numpy()
    print(numpy_array)
