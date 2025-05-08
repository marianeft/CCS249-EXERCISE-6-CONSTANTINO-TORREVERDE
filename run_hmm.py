from pos_hmm import HMM  # Import the HMM class

# Initialize and train the model
tagged_data = [
    [('The', 'DET'), ('cat', 'NOUN'), ('sleeps', 'VERB')],
    [('A', 'DET'), ('dog', 'NOUN'), ('barks', 'VERB')],
    [('My', 'DET'), ('dog', 'NOUN'), ('runs', 'VERB'), ('fast', 'ADV')],
    [('A', 'DET'), ('cat', 'NOUN'), ('meows', 'VERB'), ('loudly', 'ADV')],
    [('Your', 'DET'), ('cat', 'NOUN'), ('runs', 'VERB')],
    [('The', 'DET'), ('bird', 'NOUN'), ('sings', 'VERB'), ('sweetly', 'ADV')],
    [('A', 'DET'), ('bird', 'NOUN'), ('chirps', 'VERB')]
]

model = HMM()
model.train(tagged_data)

# Test sentences
test_sentences = [
    ["The", "can", "meows"],
    ["My", "dog", "barks", "loudly"]
]

# Run predictions
for sentence in test_sentences:
    tags = model.viterbi(sentence)
    print("Sentence:", sentence)
    print("Predicted Tags:", tags)