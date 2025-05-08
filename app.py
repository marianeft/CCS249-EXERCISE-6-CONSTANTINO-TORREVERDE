import numpy as np
from collections import defaultdict

# Given dataset
dataset = [
    "The_DET cat_NOUN sleeps_VERB",
    "A_DET dog_NOUN barks_VERB",
    "The_DET dog_NOUN sleeps_VERB",
    "My_DET dog_NOUN runs_VERB fast_ADV",
    "A_DET cat_NOUN meows_VERB loudly_ADV",
    "Your_DET cat_NOUN runs_VERB",
    "The_DET bird_NOUN sings_VERB sweetly_ADV",
    "A_DET bird_NOUN chirps_VERB"
]

# Initialize counts
transition_counts = defaultdict(lambda: defaultdict(int))
emission_counts = defaultdict(lambda: defaultdict(int))
state_counts = defaultdict(int)
vocab = set()

# Parse dataset
for sentence in dataset:
    words_tags = sentence.split()
    prev_tag = None

    for word_tag in words_tags:
        word, tag = word_tag.rsplit('_', 1)
        emission_counts[tag][word] += 1
        state_counts[tag] += 1
        vocab.add(word)

        if prev_tag:
            transition_counts[prev_tag][tag] += 1

        prev_tag = tag

# Laplace Smoothing Parameter
alpha = 1.0

# Convert counts to probabilities with smoothing
states = list(state_counts.keys())
V = len(vocab)  # Vocabulary size

transition_probs = {
    tag: {
        next_tag: (count + alpha) / (state_counts[tag] + alpha * len(states))
        for next_tag, count in transitions.items()
    }
    for tag, transitions in transition_counts.items()
}

emission_probs = {
    tag: {
        word: (count + alpha) / (state_counts[tag] + alpha * V)
        for word, count in emissions.items()
    }
    for tag, emissions in emission_counts.items()
}

# Ensure all states have nonzero probabilities for unseen words/tags
for tag in states:
    for other_tag in states:
        transition_probs[tag].setdefault(other_tag, alpha / (state_counts[tag] + alpha * len(states)))
    for word in vocab:
        emission_probs[tag].setdefault(word, alpha / (state_counts[tag] + alpha * V))

# Viterbi Algorithm using Log Probabilities
def viterbi(observed_sentence, states, transition_probs, emission_probs):
    n = len(observed_sentence)
    dp = np.full((len(states), n), -np.inf)  # Log probabilities
    backpointer = np.zeros((len(states), n), dtype=int)

    # Initialize DP table
    for i, state in enumerate(states):
        dp[i, 0] = np.log(emission_probs.get(state, {}).get(observed_sentence[0], alpha / (state_counts[state] + alpha * V)))

    # Fill DP table
    for t in range(1, n):
        for i, state in enumerate(states):
            max_prob, max_state = max(
                (dp[j, t-1] + np.log(transition_probs.get(states[j], {}).get(state, alpha / (state_counts[states[j]] + alpha * len(states)))) +
                 np.log(emission_probs.get(state, {}).get(observed_sentence[t], alpha / (state_counts[state] + alpha * V))), j)
                for j in range(len(states))
            )
            dp[i, t] = max_prob
            backpointer[i, t] = max_state

    # Backtracking
    best_path = []
    best_last_state = np.argmax(dp[:, -1])
    best_path.append(states[best_last_state])

    for t in range(n-1, 0, -1):
        best_last_state = backpointer[best_last_state, t]
        best_path.append(states[best_last_state])

    return best_path[::-1]

# Test sentences
test_sentences = [
    ["The", "can", "meows"],
    ["My", "dog", "barks", "loudly"]
]

# Run Viterbi Algorithm for each sentence
for sentence in test_sentences:
    predicted_tags = viterbi(sentence, states, transition_probs, emission_probs)
    print(list(zip(sentence, predicted_tags)))