import re

import nltk
from nltk.corpus import stopwords


# Permet de tokenizer un text
def tokenizer(text):
    # Les stopword les plus courants en anglais (is, and,...)
    nltk.download('stopwords')
    stop = stopwords.words('english')

    # Supprime les indentations HTML
    text = re.sub('<[^>]*>', '', text)
    # Enregistre toutes les émoticones
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text.lower())
    # '\W' retire tous les non mots + passe tout en miniscule + ajoute les emoticones sans le nez - (:-) -> :) )
    text = re.sub('[\W]+', ' ', text.lower()) + \
           ' '.join(emoticons).replace('-', '')
    # tokenize tous les mots en les séparant par des espaces
    tokenized = [w for w in text.split() if w not in stop]
    return tokenized

# Test sur la fonction de tokenizing
# res = tokenizer("</a>This :) is :( a test :-)!")
# print(res)
