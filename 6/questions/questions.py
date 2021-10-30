import nltk
import sys
import os
import string
import math

FILE_MATCHES = 1
SENTENCE_MATCHES = 1


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python questions.py corpus")

    # Calculate IDF values across files
    files = load_files(sys.argv[1])
    file_words = {
        filename: tokenize(files[filename])
        for filename in files
    }
    file_idfs = compute_idfs(file_words)
    
    # Prompt user for query
    query = set(tokenize(input("Query: ")))

    # Determine top file matches according to TF-IDF
    filenames = top_files(query, file_words, file_idfs, n=FILE_MATCHES)

    # Extract sentences from top files
    sentences = dict()
    for filename in filenames:
        for passage in files[filename].split("\n"):
            for sentence in nltk.sent_tokenize(passage):
                tokens = tokenize(sentence)
                if tokens:
                    sentences[sentence] = tokens

    # Compute IDF values across sentences
    idfs = compute_idfs(sentences)

    # Determine top sentence matches
    matches = top_sentences(query, sentences, idfs, n=SENTENCE_MATCHES)
    for match in matches:
        print(match)


def load_files(directory):
    """
    Given a directory name, return a dictionary mapping the filename of each
    `.txt` file inside that directory to the file's contents as a string.
    """
    mapping = dict()
    for file in os.listdir(directory):
        with open(os.path.join(directory, file)) as f:
            next(f)
            mapping[file] = f.read()
        f.close()
    return mapping
    raise NotImplementedError


def tokenize(document):
    """
    Given a document (represented as a string), return a list of all of the
    words in that document, in order.

    Process document by coverting all words to lowercase, and removing any
    punctuation or English stopwords.
    """
    words = nltk.word_tokenize(document)

    for idx in range(len(words)):
        words[idx] = words[idx].lower()

        # remove all the punctuations in string.punctuation
        if words[idx] in string.punctuation:
            words[idx] = 0

        # remove all the stopwords
        if words[idx] in nltk.corpus.stopwords.words("english"):
            words[idx] = 0

        # remove all the other words containing no English letters
        is_remove = True
        if words[idx] != 0:
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


def compute_idfs(documents):
    """
    Given a dictionary of `documents` that maps names of documents to a list
    of words, return a dictionary that maps words to their IDF values.

    Any word that appears in at least one of the documents should be in the
    resulting dictionary.
    """
    # store every unique word in a set
    words = []
    for content in documents.values():
        words += content
    words = set(words)
    
    idfs = dict()
    idfs = idfs.fromkeys(words)
    for word in idfs:
        # count the number of documents in which each word appears
        count = 0
        for content in documents.values():
            if word in content:
                count += 1
        # calculate idf
        idfs[word] = math.log(len(documents)/count)

    return idfs
    raise NotImplementedError


def top_files(query, files, idfs, n):
    """
    Given a `query` (a set of words), `files` (a dictionary mapping names of
    files to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the filenames of the the `n` top
    files that match the query, ranked according to tf-idf.
    """
    tf_idf = dict()
    tf_idf = tf_idf.fromkeys(files.keys(),0)

    for file in files:
        for word in query:
            tf_idf[file] += files[file].count(word) * idfs[word]

    top = [k for k in sorted(tf_idf.keys(), key=lambda k: tf_idf[k], reverse=True)]                
    return top[:n]

    raise NotImplementedError

    
def top_sentences(query, sentences, idfs, n):
    """
    Given a `query` (a set of words), `sentences` (a dictionary mapping
    sentences to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the `n` top sentences that match
    the query, ranked according to idf. If there are ties, preference should
    be given to sentences that have a higher query term density.
    """
    # idf_sum refers to matching word measure
    idf_sum = dict()
    idf_sum = idf_sum.fromkeys(sentences.keys(),0)
    
    # qtd refers to query term density
    qtd = dict()
    qtd = qtd.fromkeys(sentences.keys(),0)
    
    for sentence in sentences:
        qtd_count = 0
        
        for word in query:
            if word in sentences[sentence]:
                idf_sum[sentence] += idfs[word]
                qtd_count += 1
        qtd[sentence] = qtd_count/len(sentence)
        
    top = [k for k in sorted(idf_sum.keys(), key=lambda k: (idf_sum[k], qtd[k]), reverse=True)]
    return top[:n]
           
    raise NotImplementedError


if __name__ == "__main__":
    main()
