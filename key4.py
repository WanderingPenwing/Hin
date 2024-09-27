from gensim.summarization import keywords

text_en = ('Characterization of SiC MOSFET using double pulse test method.')

print(keywords(text_en,words = 5,scores = True, lemmatize = True))
