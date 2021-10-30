import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 100000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """

    prob = dict()
    prob = prob.fromkeys(corpus.keys(), 0)
    
    for each in prob.keys():
        if len(corpus[page])==0:
            prob[each] += 1/len(prob)
        else:
            prob[each] += (1-damping_factor)/len(prob)

    for linked_page in corpus[page]:
        prob[linked_page] += damping_factor/len(corpus[page])
    
    return prob

    raise NotImplementedError


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """

    pagerank = dict()
    pagerank = pagerank.fromkeys(corpus.keys(), 0)
    starting_page = random.choice(list(corpus.keys()))
    
    for i in range(n):
        prob = transition_model(corpus, starting_page, damping_factor)
        
        population = [i for i in prob.keys()]
        weight = [i for i in prob.values()]
        choice = random.choices(population, weights=weight, k=1)[0]
        pagerank[choice] += 1
        
        starting_page = choice

    for page in pagerank.keys():
        pagerank[page] /= n

    return pagerank
    raise NotImplementedError


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """

    pagerank = dict()
    pagerank = pagerank.fromkeys(corpus.keys(), 1/len(corpus.keys()))

    # pagerank_previous is used for calculate the difference between two adjacent iterations
    pagerank_temp = dict()
    pagerank_temp = pagerank.fromkeys(corpus.keys(), 1/len(corpus.keys()))
    stop = False
    
    A = (1-damping_factor)/len(corpus.keys())
    
    while not stop:
        stop = True
        for p in pagerank.keys():
            pagerank_temp[p] = A
            for i in pagerank.keys():
                if len(corpus[i])==0:
                    # if there is no link from a page, then we have to choose a new page from the corpus
                    pagerank_temp[p] += damping_factor * pagerank[i] / len(corpus.keys())
                elif p in corpus[i]:
                    # first store the new value into pagerank_temp so that we can compare the difference between two iterations
                    pagerank_temp[p] += damping_factor * pagerank[i] / len(corpus[i])
            if abs(pagerank_temp[p] - pagerank[p]) >= 0.000001:
                stop = False
                # if all differences are small enough, we can end the while loop
            pagerank[p] = pagerank_temp[p]
            
    return pagerank
    raise NotImplementedError


if __name__ == "__main__":
    main()
