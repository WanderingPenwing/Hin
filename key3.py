import spacy
nlp = spacy.load("en_core_sci_lg")
text = "Characterization of SiC MOSFET using double pulse test method"
doc = nlp(text)
print(doc.ents)
