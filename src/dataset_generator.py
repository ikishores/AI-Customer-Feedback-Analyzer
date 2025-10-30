import random
import csv
from datetime import datetime, timedelta

positive_feedback = [
    "The product is amazing! It works perfectly.",
    "Excellent service and quick response.",
    "Very happy with the purchase, thank you!",
    "Great experience, will recommend to others.",
    "Loved it! Totally worth the money."
]

neutral_feedback = [
    "The product is okay, not bad.",
    "Average experience overall.",
    "It works fine, nothing special.",
    "No major issues but can be better.",
    "Neutral review, it’s acceptable."
]

negative_feedback = [
    "Very bad experience. Not satisfied at all.",
    "Product arrived damaged and late.",
    "Customer support was unhelpful.",
    "Terrible service, waste of money.",
    "Not happy with the quality."
]

def generate_feedback(n=1500):
    data = []
    for i in range(n):
        sentiment = random.choice(["positive", "neutral", "negative"])
        if sentiment == "positive":
            text = random.choice(positive_feedback)
        elif sentiment == "neutral":
            text = random.choice(neutral_feedback)
        else:
            text = random.choice(negative_feedback)
        date = datetime.now() - timedelta(days=random.randint(0, 180))
        data.append([i+1, date.strftime("%Y-%m-%d"), text, sentiment])
    return data

data = generate_feedback()

with open("../data/simulated_feedback.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["id", "date", "text", "sentiment_label"])
    writer.writerows(data)

print(" Generated simulated_feedback.csv with 1500 records!")
