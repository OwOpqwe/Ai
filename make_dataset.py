import json
import os

os.makedirs("data", exist_ok=True)

facts = {
    "name": "Alex",
    "birthday": "June 15",
    "height": "170 cm",
    "weight": "60 kg",
    "shoe_size": "EU 42",
    "favorite_subject": "Mathematics",
    "hobby": "basketball",
    "future_goal": "software engineer"
}

questions = {
    "name": [
        "What is my name?",
        "Who am I?",
        "Tell me my name.",
        "What name do I use?",
        "What am I called?",
        "Remind me of my name.",
        "Do you know my name?",
        "Can you identify me?"
    ],

    "birthday": [
        "When is my birthday?",
        "What day is my birthday?",
        "Tell me my birthday.",
        "When was I born?",
        "Do you know my birthday?",
        "What is my birth date?",
        "Remind me when my birthday is.",
        "When do I celebrate my birthday?"
    ],

    "height": [
        "What is my height?",
        "How tall am I?",
        "Tell me my height.",
        "Do you know how tall I am?",
        "What is my height in centimeters?",
        "Remind me how tall I am.",
        "How many centimeters tall am I?",
        "Can you tell me my height?"
    ],

    "weight": [
        "What is my weight?",
        "How much do I weigh?",
        "Tell me my weight.",
        "Do you know how much I weigh?",
        "What is my weight in kilograms?",
        "Remind me of my weight.",
        "How many kilograms do I weigh?",
        "Can you tell me my weight?"
    ],

    "shoe_size": [
        "What is my shoe size?",
        "Which shoe size do I wear?",
        "Tell me my shoe size.",
        "Do you know my shoe size?",
        "What size are my shoes?",
        "Remind me of my shoe size.",
        "What is my footwear size?",
        "What size shoes do I wear?"
    ],

    "favorite_subject": [
        "What is my favorite subject?",
        "Which subject do I like most?",
        "Tell me my favorite class.",
        "What subject do I enjoy?",
        "Do you know my favorite school subject?",
        "Which class is my favorite?",
        "What subject do I like best?",
        "What is my favorite school topic?"
    ],

    "hobby": [
        "What is my hobby?",
        "What do I do for fun?",
        "Tell me about my hobby.",
        "What activity do I enjoy?",
        "What do I do in my free time?",
        "What is my favorite pastime?",
        "What do I like doing?",
        "Which activity is my hobby?"
    ],

    "future_goal": [
        "What is my future goal?",
        "What do I want to become?",
        "Tell me my career goal.",
        "What job do I want in the future?",
        "What is my future ambition?",
        "What career am I aiming for?",
        "What do I want to be?",
        "What do I hope to become?"
    ]
}

dataset = []

for category, question_list in questions.items():
    for question in question_list:
        dataset.append({
            "question": question,
            "intent": category,
            "answer": facts[category]
        })

with open("data/qa_dataset.json", "w", encoding="utf-8") as file:
    json.dump(dataset, file, indent=4)

with open("data/facts.json", "w", encoding="utf-8") as file:
    json.dump(facts, file, indent=4)

print("Dataset successfully created!")
print("Number of training examples:", len(dataset))
print("Number of personal facts:", len(facts))
