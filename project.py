import pandas as pd
from nltk.tokenize import sent_tokenize
import nltk
import ssl
from collections import Counter #utile per il dizionario
import numpy as np

PATH=r"ch2.txt"


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

#DEFINIAMO LA LISTA DELLE STOPWORDS
# Stopwords considered as noise in the text. Text may contain stop words such as is, am, are, this, a, an, the, etc.

from nltk.corpus import stopwords
nltk.download('stopwords')
stop_words= set(stopwords.words("english"))


#aggiungo alla lista di stop words anche lettere e numeri
lettere = list('abcdefghijklmnopqrstuvwxyz')
numeri = list('0123456789')

for i in range(0,25):
    stop_words.add(lettere[i])
for i in range(0,10):
    stop_words.add(numeri[i])


# REMOVING STOPWORDS
text1 = word_tokenize(raw.lower())
words_nostop = [x for x in text1 if x not in stop_words ] #parole rimanenti, che non sono
#stopwords
print(words_nostop)

# Come si puo vedere, si passa da 35528 token, a 29330
len(tokenized_word) 
len(words_nostop)


# PULISCO INOLTRE DALLA PUNTEGGIATURA
words_nopunct= [word for word in words_nostop if word.isalnum()]

len(words_nopunct)
#5609

#vediamo i risultati dopo questa prima pulizia

fdist = FreqDist(words_nopunct)
fdist.plot(30,cumulative=False)
plt.show()

###TAGGING

nltk.download('averaged_perceptron_tagger')

from nltk import jsontags
print("Parts of speech:", nltk.pos_tag(words_nopunct))

tagged_words=nltk.pos_tag(words_nopunct)

type(tagged_words) #lista di tuple ('parola', 'tag')

#lista di tag riferita a parole che vogliamo rimuovere
REM=['CC','CD','DT','IN','MD','NNP','NNPS','PRP','PSRP$','RB','RBR','RBS','TO','WDT','WPD$','WRB']

len(tagged_words) #5609
tagged_words

#la funzione restituisce un vettore lungo come la lista di tuple con 0 se la tupla è da tenere
#e 1 se è da togliere
def eliminare(tagged_words1):
   
    togli=np.zeros(len(tagged_words1))
    res = list(zip(*tagged_words1)) #zippiamo la lista di tuple
    res=res[1] #prendiamo solo i tag
    for i in np.arange(len(tagged_words1)):  
        #il ciclo prende nota di quali sono le parole da eliminare 
        tup=res[i]
        for j in REM:
            if tup==j:
                togli[i]=1
    return togli


togli=eliminare(tagged_words)
len(togli) #numero di elementi in totale
sum(togli) #numero di elementi da eliminare: 875


indici=np.zeros(int(sum(togli))) #inizializziamo


togli= np.array(togli, dtype=int)
finali=list(np.array(words_nopunct)[togli==0])

len(finali )#4734


#visualizziamo i risultati finali dopo la pulizia

fdist = FreqDist(finali)
fdist.plot(30,cumulative=False)
plt.show()



# STEMMING:
# Stemming is a process of linguistic normalization, which reduces words to their word root word or chops off the derivational affixes.
# For example, connection, connected, connecting word reduce to a common word "connect".
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize



ps = PorterStemmer()
stemmed_words=[]
for w in finali:
    stemmed_words.append(ps.stem(w))

#print("Filtered Sentence:",stopwords)
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
for w in finali:
    lemmed_words.append(lem.lemmatize(w,"v"))



# STRINGA PULITA

dict1=Counter(finali)
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

# bigrams 
bigrams = nltk.bigrams(finali)
print(finali)

fdist = FreqDist(bigrams)
fdist.plot(30,cumulative=False)
plt.show()









