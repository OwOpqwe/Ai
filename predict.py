import re
import torch
import torch.nn as nn

MAX_LENGTH = 16

def tokenize(text):
    return re.findall(r"[a-z0-9]+", text.lower())


class MyAI(nn.Module):

    def __init__(self, vocab_size, number_of_classes):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size, 48, padding_idx=0
        )

        self.network = nn.Sequential(
            nn.Linear(48, 64),
            nn.ReLU(),
            nn.Linear(64, number_of_classes)
        )

    def forward(self, x):

        vectors = self.embedding(x)

        mask = (x != 0).unsqueeze(-1)

        average = (
            (vectors * mask).sum(1)
            / mask.sum(1).clamp(min=1)
        )

        return self.network(average)


# Load saved model
def load_model():

    saved = torch.load(
        "trained_model.pt",
        map_location="cpu",
        weights_only=True
    )

    model = MyAI(
        len(saved["vocab"]),
        len(saved["intents"])
    )

    model.load_state_dict(saved["model_state"])

    model.eval()

    return model, saved


# Ask the trained AI a question
def answer_question(question, model=None, saved=None):

    if model is None:
        model, saved = load_model()

    words = tokenize(question)

    ids = [
        saved["vocab"].get(word, 1)
        for word in words
    ][:MAX_LENGTH]

    ids += [0] * (MAX_LENGTH - len(ids))

    input_tensor = torch.tensor([ids])

    with torch.no_grad():

        output = model(input_tensor)

        probabilities = torch.softmax(output, dim=1)

        confidence, prediction = probabilities.max(dim=1)

    intent = saved["intents"][prediction.item()]

    # Avoid pretending to know completely unrelated questions
    if confidence.item() < 0.35:
        return {
            "answer": "I'm not sure how to answer that question.",
            "predicted_intent": intent,
            "confidence": round(confidence.item(), 3)
        }

    fact = saved["facts"][intent]

    answers = {
        "name": f"Your name is {fact}.",
        "birthday": f"Your birthday is {fact}.",
        "height": f"Your height is {fact}.",
        "weight": f"Your weight is {fact}.",
        "shoe_size": f"Your shoe size is {fact}.",
        "favorite_subject": f"Your favorite subject is {fact}.",
        "hobby": f"Your hobby is {fact}.",
        "future_goal": f"Your future goal is to become a {fact}."
    }

    return {
        "answer": answers[intent],
        "predicted_intent": intent,
        "confidence": round(confidence.item(), 3)
    }


# Test the model
if __name__ == "__main__":

    model, saved = load_model()

    test_questions = [
        "What is my shoe size?",
        "Could you remind me how tall I am?",
        "Which subject do I like most?",
        "What do I want to become?"
    ]

    for question in test_questions:

        result = answer_question(
            question, model, saved
        )

        print("\nQuestion:", question)
        print("AI:", result["answer"])
        print("Confidence:", result["confidence"])
