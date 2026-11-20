
from fastapi import FastAPI
from pydantic import BaseModel

from model import load_trained_model, predict

# Create API
app = FastAPI(title="My Personal AI API")

# Load trained AI
model, vocabulary, answers = load_trained_model()


class Question(BaseModel):
    question: str


@app.get("/")
def home():
    return {"message": "My AI API is running!"}


@app.post("/ask")
def ask_ai(data: Question):
    response = predict(
        data.question,
        model,
        vocabulary,
        answers
    )

    return {
        "question": data.question,
        "answer": response
    }
