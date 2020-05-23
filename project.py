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

def elimina(tagged_words1):#elimina gli elementi in base al tag
    res = list(zip(*tagged_words1)) #prendo solo i tag
    len(tagged_words1)
    len(res)
    res=res[1]
    l=0
    i=0
    while (l+i<len(tagged_words1)):  #il ciclo mi rimuove le parole che non "servono
        tup=res[i]
        for j in REM:
            if tup==j: #serve il secondo elemento della tupla: il tag (il non funzionamento dipende da questo)
                tagged_words1.remove(tagged_words1[i])#remove non và bene per le tuple
                l=l+1
                i=i-1
        i=i+1
    return tagged_words1

tagged_words=elimina(tagged_words)
len(tagged_words)#dovrebbe funzionare

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

# bigrams 
bigrams = list(nltk.bigrams(final))
print(final)

fdist = FreqDist(bigrams)
plt.figure(figsize=(20, 10))
fdist.plot(30,cumulative=False)
plt.savefig( 'myfig.jpg' ) # Comando che salva l'immagine nella cartella TextMining

# Raw popularity count is too crude of a measure.
# We have to find more clever statistics to be able to pick out meaningful phrases easily.
# For a given pair of words, the method tests two hypotheses on the observed dataset.
# Hypothesis 1 (the null hypothesis) says that word 1 appears independently from word 2.
# Hypothesis 2 (the alternate hypothesis) says that seeing word 1 changes the likelihood of seeing word 2.

# Hence, the likelihood ratio test for phrase detection (a.k.a. collocation extraction) asks the following question:
# Are the observed word occurrences in a given text corpus more likely to have been generated from a model where
# the two words occur independently from one another?

import math

def logL(p,k,n):
    return k * math.log(p) + (n-k)* math.log(1-p+0.0001)

unici = list(set(bigrams)) # valori unici dei bigrammi

def log_likelihood_statistic (p,k,n):
    p1 = p[0]
    p2 = p[1]
    p = p[2]
    k1 = k[0]
    k2 = k[1]
    n1 = n[0]
    n2 = n[1]

    return 2 * ( logL(p1,k1,n1) + logL(p2,k2,n2) - logL(p,k1,n1) -logL(p,k2,n2) )


p = np.zeros(3)
ris = np.zeros(len(unici))

# Vado a calcolare la log-rapporto di verosimiglianza!!
for i in range(0,len(unici)):
    k = np.zeros(2)
    n = np.zeros(2)
    for h in range(0,len(bigrams)):

            if unici[i][0] == bigrams[h][0]:
                k[0] = k[0] + 1
                n[0] = n[0] + 1
            if unici[i][0] == bigrams[h][1]:
                n[0] = n[0] + 1

            if unici[i][1] == bigrams[h][1]:
                k[1] = k[1] + 1
                n[1] = n[1] + 1
            if unici[i][1] == bigrams[h][0]:
                n[1] = n[1] + 1

    p[0] = (k[0] / n[0])
    p[1] = (k[1] / n[1])
    p[2] = (k[0] + k[1]) / (n[0] + n[1])

    ris[i] = log_likelihood_statistic(p,k,n)

ris = list(ris)
log_ordinata= sorted(range(len(ris)), key=lambda k: ris[k])  #OCCHIO AL MENO !


for k in range(0,30):
  print(unici[log_ordinata[k]])


