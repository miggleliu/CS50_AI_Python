import nltk
import sys

TERMINALS = """
Adj -> "country" | "dreadful" | "enigmatical" | "little" | "moist" | "red"
Adv -> "down" | "here" | "never"
Conj -> "and" | "until"
Det -> "a" | "an" | "his" | "my" | "the"
N -> "armchair" | "companion" | "day" | "door" | "hand" | "he" | "himself"
N -> "holmes" | "home" | "i" | "mess" | "paint" | "palm" | "pipe" | "she"
N -> "smile" | "thursday" | "walk" | "we" | "word"
P -> "at" | "before" | "in" | "of" | "on" | "to"
V -> "arrived" | "came" | "chuckled" | "had" | "lit" | "said" | "sat"
V -> "smiled" | "tell" | "were"
"""

NONTERMINALS = """
S -> NP VP | S Conj S
NP -> N | AdjP N | Det N | Det AdjP N | NP PP 
VP -> V | V NP | V P NP | V AdvP P NP | AdvP VP | VP AdvP | VP Conj VP | VP VP Conj VP | VP Conj VP Conj VP
AdjP -> Adj | Adj Adj | Adj Adj Adj | Adj Conj Adj | Adj Adj Conj Adj | Adj Conj Adj Conj Adj
PP -> P NP | PP PP | PP Conj PP
AdvP -> Adv | Adv Conj Adv | Adv Adv Conj Adv
"""

grammar = nltk.CFG.fromstring(NONTERMINALS + TERMINALS)
parser = nltk.ChartParser(grammar)


def main():

    # If filename specified, read sentence from file
    if len(sys.argv) == 2:
        with open(sys.argv[1]) as f:
            s = f.read()

    # Otherwise, get sentence as input
    else:
        s = input("Sentence: ")

    # Convert input into list of words
    s = preprocess(s)
    
    # Attempt to parse sentence
    try:
        trees = list(parser.parse(s))
    except ValueError as e:
        print(e)
        return
    if not trees:
        print("Could not parse sentence.")
        return

    # Print each tree with noun phrase chunks
    for tree in trees:
        tree.pretty_print()
        
        print("Noun Phrase Chunks")
        for np in np_chunk(tree):
            print(" ".join(np.flatten()))


def preprocess(sentence):
    """
    Convert `sentence` to a list of its words.
    Pre-process sentence by converting all characters to lowercase
    and removing any word that does not contain at least one alphabetic
    character.
    """
    words = nltk.word_tokenize(sentence)
    
    for idx in range(len(words)):
        words[idx] = words[idx].lower()
          
        is_remove = True
        for letter in words[idx]:
            # The ASCII value for 'a' and 'z' are 97 and 122 respectively
            if 97 <= ord(letter) <= 122:
                is_remove = False
                
        if is_remove:
            words[idx] = 0

    while True:
        try:
            words.remove(0)
        except ValueError:
            break
            
    return words
    raise NotImplementedError


def np_chunk(tree):
    """
    Return a list of all noun phrase chunks in the sentence tree.
    A noun phrase chunk is defined as any subtree of the sentence
    whose label is "NP" that does not itself contain any other
    noun phrases as subtrees.
    """
    chunks = []
    is_np = False
    
    for subtree in tree.subtrees(filter=lambda x: x.label()=='NP'):
        is_np = True
        for sub_subtree in subtree.subtrees():
            if sub_subtree.height() == subtree.height():
                # In this case the sub_subtree is the subtree itself
                continue
            if sub_subtree.label() == 'NP':
                is_np = False
                break
                
        chunks.append(subtree) if is_np else None
        
    return chunks
    raise NotImplementedError


if __name__ == "__main__":
    main()
