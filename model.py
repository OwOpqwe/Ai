
import re
import torch
import torch.nn as nn


def tokenize(text):
    return re.findall(r"[a-z0-9]+", text.lower())


class QuestionModel(nn.Module):

    def __init__(self, input_size, output_size):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_size)
        )

    def forward(self, x):
        return self.network(x)


def vectorize(question, vocabulary):
    words = tokenize(question)
    vector = torch.zeros(len(vocabulary))

    for word in words:
        if word in vocabulary:
            vector[vocabulary[word]] = 1.0

    return vector


def load_trained_model():

    checkpoint = torch.load(
        "trained_model.pth",
        map_location="cpu",
        weights_only=True
    )

    vocabulary = checkpoint["vocabulary"]
    answers = checkpoint["answers"]

    model = QuestionModel(
        len(vocabulary),
        len(answers)
    )

    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    return model, vocabulary, answers


def predict(question, model, vocabulary, answers):

    vector = vectorize(question, vocabulary)

    with torch.no_grad():
        prediction = model(vector.unsqueeze(0))
        probabilities = torch.softmax(prediction, dim=1)

        confidence, predicted_class = torch.max(
            probabilities, dim=1
        )

    if confidence.item() < 0.35:
        return "I don't know the answer to that question yet."

    return answers[predicted_class.item()]
