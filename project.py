import pandas as pd
from nltk.tokenize import sent_tokenize
import nltk
import ssl
from collections import Counter #utile per il dizionario
import numpy as np

PATH=r"C:\Users\Francesco\Desktop\altro\ch2.txt"




def Convert(raw1): #converte la stringa in lista (al momento non utile ma potrebbe sempre servire)
    raw1=raw1.strip("\n") 
    raw1 = raw1.replace('\n'," ")
    l=raw1.split(" ")
    return l


# CARICAMENTO TXT IN FORMATO STRINGA
f = f = open(PATH,encoding="utf8")#questo dovrebbe funzionare per tutti
raw = f.read()
raw[1:200]#prova
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


# ELIMINAZIONE  le lettere e i numeri
list('abcdefghijklmnopqrstuvwxyz')
lettere = list('abcdefghijklmnopqrstuvwxyz')
numeri = list('0123456789')

for i in range(0,25):
    stop_words.add(lettere[i])
for i in range(0,10):
    stop_words.add(numeri[i])


# These are the several step words considered!
print(stop_words)

# REMOVING STOPWORDS
text1 = word_tokenize(raw.lower())
print(text1)
stopwords = [x for x in text1 if x not in stop_words ]
print(stopwords)

# Come si puo vedere, si passa da 35528 token, a 30071

# Rivediamo graficamente ora i risultati dopo la pulizia
fdist = FreqDist(stopwords)
fdist.plot(30,cumulative=False)
plt.show()


###TAGGING

nltk.download('averaged_perceptron_tagger')

from nltk import jsontags
print("Parts of speech:", nltk.pos_tag(tokenized_word))

tagged_words=nltk.pos_tag(tokenized_word)

type(tagged_words) #lista di tuple ('parola', 'tag')

#lista di tag riferita a parole che voglio rimuovere
REM=['CC','CD','DT','IN','MD','NNP','NNPS','PRP','PSRP$','RB','RBR','RBS','TO','WDT','WPD$','WRB']

len(tagged_words)


tagged_words[1]
#########questo non va bisogna vedere soltanto come rimuovere gli elementi di una tupla

tagged_words1=tagged_words
res = list(zip(*tagged_words)) #prendo solo i tag
len(res[1])
for i in np.arange(0,35527):  
    tup=res[i]
    for j in REM:
        if tup[i]==j: #serve il secondo elemento della tupla: il tag (il non funzionamento dipende da questo)
            tagged_words1=tagged_words[i].remove()#remove non và bene per le tuple
            
    
len(tagged_words1)
####adesso invece anche questo dà errore




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
type(final)

final[45:100] #prova

# STRINGA PULITA

dict1=Counter(final)
vv={k: v for k, v in sorted(dict1.items(), key=lambda item: item[1])}

#tramite i comandi che seguono possiamo modellare il nostro dizionario eliminando facilmente elementi
dict1 = dict((k, v) for k, v in dict1.items() if v >= 10)
dict2=dict((k, v) for k, v in dict1.items() if v <= 100)#potrebbe essere interessante togliere anche le parole che si ripetono troppo
dict2

# Rivediamo graficamente ora i risultati dopo la pulizia
fdist = FreqDist(dict2) #si puo fare sia su una stringa che su un dizionario
fdist.plot(30,cumulative=False)
plt.show()
print ("fatto")










