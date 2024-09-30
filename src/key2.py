from sklearn.feature_extraction.text import TfidfVectorizer
from itertools import combinations

vectorizer = TfidfVectorizer(stop_words='english')

def create_queries(subject) :

    print("\n### Getting Keywords ###\n")

    tfidf_matrix = vectorizer.fit_transform([subject])

    feature_names = vectorizer.get_feature_names_out()
    
    print("* Preparation done")

    sorted_indices = tfidf_matrix[0].toarray()[0].argsort()[::-1]

    text_keywords = []

    for i in range(5):  # Change 3 to however many keywords you want
        if i < len(sorted_indices):
            text_keywords.append(feature_names[sorted_indices[i]])

    log = f"## Keywords\n\n{text_keywords}\n\n"

    queries = []

    for r in range(1, len(text_keywords) + 1):
        comb = combinations(text_keywords, r)
        queries.extend(comb)

    final_queries = [subject] + ["\"" + "\" OR \"".join(query) + "\"" for query in queries]

    #final_queries.ins(subject)

    print("* query generated")

    return final_queries, log
