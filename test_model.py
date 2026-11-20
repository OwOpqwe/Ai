
from model import load_trained_model, predict


# Load the saved AI model
print("Loading trained AI...")

model, vocabulary, answers = load_trained_model()

print("AI loaded successfully!")


# Questions to test
questions = [
    "What is my height?",
    "How tall am I?",
    "What is my shoe size?",
    "Which subject do I like most?",
    "What is my hobby?",
    "Could you remind me how tall I am?"
]


# Test predictions
print("\n===== AI TEST RESULTS =====")

for question in questions:

    response = predict(
        question,
        model,
        vocabulary,
        answers
    )

    print("\nQuestion:", question)
    print("AI Answer:", response)

print("\nTesting complete!")
