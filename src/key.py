from keybert import KeyBERT
from itertools import combinations

def create_queries(subject) :

    print("\n### Getting Keywords ###\n")

    kw_model = KeyBERT()

    print("* Got Keybert")

    keywords = kw_model.extract_keywords(subject, keyphrase_ngram_range=(1, 2), stop_words='english', use_mmr=True, diversity=0.7)

    print("* keywords extracted")
    
    sorted_keywords = sorted(keywords, key=lambda x: -x[1])
    text_keywords = [x[0] for x in sorted_keywords]

    log = f"## Keywords\n\n{text_keywords}\n\n"

    queries = []

    for r in range(1, len(text_keywords) + 1):
        comb = combinations(text_keywords, r)
        queries.extend(comb)

    final_queries = [subject] + ["\"" + "\" OR \"".join(query) + "\"" for query in queries]

    #final_queries.ins(subject)

    print("* query generated")

    return final_queries, log
