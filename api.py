from fastapi import FastAPI
from pydantic import BaseModel

from predict import load_model, answer_question

app = FastAPI(
    title="Alex AI API",
    description="API for my own trained AI model"
)

# Load the trained model when the API starts
model, saved = load_model()


class Question(BaseModel):
    question: str


@app.get("/")
def home():

    return {
        "message": "Alex AI API is running!"
    }


@app.post("/ask")
def ask(data: Question):

    result = answer_question(
        data.question,
        model,
        saved
    )

    return result
