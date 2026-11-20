
import json
import torch
import torch.nn as nn
from collections import Counter

from model import QuestionModel, tokenize, vectorize


# 1. Load training dataset
with open("dataset.json", "r", encoding="utf-8") as file:
    data = json.load(file)

print("Dataset loaded:", len(data), "examples")


# 2. Create vocabulary
all_words = []

for item in data:
    all_words.extend(tokenize(item["question"]))

word_counts = Counter(all_words)

vocabulary = {
    word: index
    for index, word in enumerate(word_counts.keys())
}


# 3. Create answer categories
answers = sorted(set(item["answer"] for item in data))

answer_to_index = {
    answer: index
    for index, answer in enumerate(answers)
}


# 4. Convert questions into numbers
X = torch.stack([
    vectorize(item["question"], vocabulary)
    for item in data
])

Y = torch.tensor([
    answer_to_index[item["answer"]]
    for item in data
], dtype=torch.long)


# 5. Create neural network
model = QuestionModel(
    len(vocabulary),
    len(answers)
)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.01
)


# 6. Train neural network
epochs = 500

print("\nStarting AI training...\n")

for epoch in range(epochs):

    model.train()

    optimizer.zero_grad()

    predictions = model(X)

    loss = criterion(predictions, Y)

    loss.backward()

    optimizer.step()

    if (epoch + 1) % 50 == 0:
        print(
            f"Epoch [{epoch + 1}/{epochs}] "
            f"Loss: {loss.item():.4f}"
        )


# 7. Save trained model
torch.save({
    "model_state": model.state_dict(),
    "vocabulary": vocabulary,
    "answers": answers
}, "trained_model.pth")


print("\nTraining complete!")
print("Model saved as trained_model.pth")
