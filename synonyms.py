import nltk
from nltk.corpus import wordnet

# Download WordNet data (run once)
nltk.download('wordnet', quiet=True)

def get_all_synonyms(word, pos=None):
    """
    Retrieve all synonyms for a given word using WordNet.
    Args:
        word (str): The word to find synonyms for.
        pos (str, optional): Part of speech (e.g., 'n' for noun, 'v' for verb).
    Returns:
        list: List of unique synonyms.
    """
    synonyms = set()
    # Get all synsets for the word, optionally filtered by part of speech
    synsets = wordnet.synsets(word, pos=pos)
    for synset in synsets:
        for lemma in synset.lemmas():
            synonym = lemma.name().replace('_', ' ')
            synonyms.add(synonym)
    return sorted(list(synonyms)) if synonyms else [word]  # Return original word if no synonyms

def main():
    # Get user input
    word = input("Enter a word to find its synonyms (e.g., 'explain'): ").strip().lower()
    
    if not word:
        print("Error: Please enter a valid word.")
        return
    
    # Retrieve synonyms (focus on verbs for programming context, but can be changed)
    synonyms = get_all_synonyms(word, pos=wordnet.VERB)
    
    # Print results
    print(f"\nSynonyms for '{word}' (as a verb):")
    if synonyms == [word]:
        print(f"No synonyms found for '{word}'.")
    else:
        for i, synonym in enumerate(synonyms, 1):
            print(f"{i}. {synonym}")

if __name__ == '__main__':
    main()