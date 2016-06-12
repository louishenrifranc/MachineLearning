import re


# Permet de tokenizer un text
def tokenizer(text):
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
