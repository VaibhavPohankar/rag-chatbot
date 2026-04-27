import re
from difflib import SequenceMatcher

# Load data
with open("data/sample.txt", "r", encoding="utf-8") as f:
    lines = [line.strip() for line in f.readlines() if line.strip()]

print("✅ Ready\n")

# Stop words
stop_words = {"what", "is", "the", "a", "an", "of", "in", "to"}

# Similarity function for fuzzy matching
def similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()

def retrieve_context(query):
    # Clean query
    query_words = set(re.findall(r'\b\w+\b', query.lower())) - stop_words

    best_match = None
    max_score = 0

    for line in lines:
        line_words = set(re.findall(r'\b\w+\b', line.lower()))

        score = 0
        for q in query_words:
            for w in line_words:
                if similarity(q, w) > 0.8:  # fuzzy match threshold
                    score += 1

        if score > max_score:
            max_score = score
            best_match = line

    if max_score == 0:
        return None

    return best_match


while True:
    query = input("Ask: ")
    if query.lower() in ["exit", "quit"]:
        break

    context = retrieve_context(query)

    if context:
        print(f"Matched Context: {context}")
        print("Answer:", context)
    else:
        print("Answer: I don't know")