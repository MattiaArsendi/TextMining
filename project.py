import pandas as pd
from nltk.tokenize import sent_tokenize
import nltk
import ssl

# CARICAMENTO TXT IN FORMATO STRINGA
f = open('ch2.txt')
raw = f.read()

# FIRST STEP, TOKENIZATION, namely the process of breaking down a text paragraph into smaller chunks
# What's a token?  Token is a single entity that is building blocks for sentence or paragraph.
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
nltk.download("punkt")

#SENTENCE TOKENIZATION
from nltk.tokenize import sent_tokenize
tokenized_text=sent_tokenize(raw)
print(tokenized_text)

# WORD TOKENIZATION
from nltk.tokenize import word_tokenize
tokenized_word=word_tokenize(raw)
print(tokenized_word)

# Frequency distribution
from nltk.probability import FreqDist
import matplotlib.pyplot as plt

fdist = FreqDist(tokenized_word)
fdist.plot(30,cumulative=False)
plt.show()


# Stopwords considered as noise in the text. Text may contain stop words such as is, am, are, this, a, an, the, etc.

from nltk.corpus import stopwords
nltk.download('stopwords')
stop_words= set(stopwords.words("english"))
# These are the several step words considered!
print(stop_words)

# REMOVING STOPWORDS
text1 = word_tokenize(raw.lower())
print(text1)
stopwords = [x for x in text1 if x not in a]
print(stopwords)

# Come si puo vedere, si passa da 35528 token, a 30071

# Rivediamo graficamente ora i risultati dopo la pulizia
fdist = FreqDist(stopwords)
fdist.plot(30,cumulative=False)
plt.show()

# STEMMING:
# Stemming is a process of linguistic normalization, which reduces words to their word root word or chops off the derivational affixes.
# For example, connection, connected, connecting word reduce to a common word "connect".
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize

ps = PorterStemmer()
stemmed_words=[]
for w in stopwords:
    stemmed_words.append(ps.stem(w))

print("Filtered Sentence:",stopwords)
print("Stemmed Sentence:",stemmed_words)

#LEMMAIZATION:
# Lemmatization reduces words to their base word, which is linguistically correct lemmas.
# Lemmatization is usually more sophisticated than stemming.
# Stemmer works on an individual word without knowledge of the context. For example, The word "better" has "good" as its lemma.
# This thing will miss by stemming because it requires a dictionary look-up.

from nltk.stem.wordnet import WordNetLemmatizer
nltk.download('wordnet')
lem = WordNetLemmatizer()


lemmed_words=[]
for w in stopwords:
    lemmed_words.append(lem.lemmatize(w,"v"))

# PULISCO INOLTRE DALLA PUNTEGGIATURA
final= [word for word in lemmed_words if word.isalnum()]


# STRINGA PULITA
print(final)


# Rivediamo graficamente ora i risultati dopo la pulizia
fdist = FreqDist(final)
fdist.plot(30,cumulative=False)
plt.show()










