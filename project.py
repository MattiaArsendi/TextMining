import nltk
import ssl
from collections import Counter #utile per il dizionario
import numpy as np


try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
nltk.download("punkt")

#######################TOKENIZATION
from nltk.tokenize import word_tokenize



##################DEFINIAMO LA LISTA DELLE STOPWORDS
# Stopwords considered as noise in the text. Text may contain stop words such as is, am, are, this, a, an, the, etc.

from nltk.corpus import stopwords
nltk.download('stopwords')
stop_words= set(stopwords.words("english"))

# ELIMINAZIONE lettere e numeri
list('abcdefghijklmnopqrstuvwxyz')
lettere = list('abcdefghijklmnopqrstuvwxyz')
numeri = list('0123456789')

for i in range(0,25):
    stop_words.add(lettere[i])
for i in range(0,10):
    stop_words.add(numeri[i])


stop_words.add("oo")
stop_words.add("ooo")
stop_words.add("oooo")
stop_words.add("ooooo")
stop_words.add("oooooo")
stop_words.add("ooooooo")
stop_words.add("oooooooo")
stop_words.add("ooooooooo")
stop_words.add("pr")
stop_words.add("fr")
stop_words.add("f0")
stop_words.add("f1")
stop_words.add("f2")
stop_words.add("pn")
stop_words.add("et")
stop_words.add("al")
stop_words.add("panel")
stop_words.add("figure")
stop_words.add("err")

#######################################TAGGING

nltk.download('averaged_perceptron_tagger')

#lista di tag riferita a parole che vogliamo rimuovere
REM=['CC','CD','DT','IN','MD','NNP','NNPS','PRP','PSRP$','RB','RBR','RBS','TO','WDT','WPD$','WRB']

#la funzione restituisce un vettore lungo come la lista di tuple con 0 se la tupla è da tenere e 1 se è da togliere
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


######################################LEMMAIZATION:
# Lemmatization reduces words to their base word, which is linguistically correct lemmas.
# Lemmatization is usually more sophisticated than stemming.
# Stemmer works on an individual word without knowledge of the context. For example, The word "better" has "good" as its lemma.
# This thing will miss by stemming because it requires a dictionary look-up.

from nltk.stem.wordnet import WordNetLemmatizer
nltk.download('wordnet')
lem = WordNetLemmatizer()



#######################CERCHIAMO I BIGRAMMI IMPORTANTI PER OGNI CAPITOLO


import tqdm
from nltk.probability import FreqDist


PATH=["ch2.txt", "ch3.txt", "ch4.txt", "ch5.txt","ch6.txt","ch7.txt",
        "ch8.txt","ch9.txt","ch10.txt","ch11.txt","ch12.txt","ch13.txt",
         "ch14.txt", "ch15.txt","ch16.txt","ch17.txt","ch18.txt"]

d=[]
l_cap = [] # Lunghezza di ogni capitolo
for i in tqdm.tqdm(range(0, len(PATH))):
    perc=PATH[i]
    f = open(perc,encoding="utf8")
    raw = f.read()
    #############################PULIZIA
    #TOKENIZATION
    tokenized_word=word_tokenize(raw) 
    text1 = word_tokenize(raw.lower())
    #RIMOZIONE DELLE STOPWORDS
    stopwords = [x for x in text1 if x not in stop_words ] 
    #RIMOZIONE DELLA PUNTEGGIATURA
    words_nopunct= [word for word in stopwords if word.isalnum()] 
    #TAGGING
    tagged_words=nltk.pos_tag(words_nopunct) 
    togli=eliminare(tagged_words)
    indici=np.zeros(int(sum(togli))) 
    togli= np.array(togli, dtype=int)
    finali=list(np.array(words_nopunct)[togli==0])
    #LEMMING
    lemmed_words=[]
    for w in finali:
        lemmed_words.append(lem.lemmatize(w,"v"))
    finali=lemmed_words
    #ESTRAZIONE DEI BIGRAMMI E LORO DISTRIBUZIONE DI FREQUENZA
    bigrams = list(nltk.bigrams(finali))
    l_cap.append(len(bigrams))
    fdist = FreqDist(bigrams)
    ####################ESTRAZIONE DEI BIGRAMMI PIU FREQUENTI
    freqtot=len(bigrams)*0.02
    d1=0
    c1=15
    while(d1<freqtot):
        dict2=dict((k, v) for k, v in fdist.items() if v >= c1)
        d1=sum(dict2.values())
        c1=c1-1
    d.append(dict2)

#contiamo il numero totale di bigrammi estratti    


d[1][('linear', 'method')] = d[1].pop(('linear', 'methods'))
d[1][('method', 'regression')] = d[1].pop(('methods', 'regression'))
d[1][('principal', 'component')] = d[1].pop(('principal', 'components'))
d[2][('linear', 'method')] = d[2].pop(('linear', 'methods'))
d[2][('method', 'classiﬁcation')] = d[2].pop(('methods', 'classiﬁcation'))
del d[3][('smooth', 'splines')]
d[3][('smooth', 'spline')]  =  52
d[4][('smooth', 'method')] = d[4].pop(('smooth', 'methods'))
del d[12][('principal', 'components')]
d[12][('principal', 'component')]  =  83
del d[16][('principal', 'components')]
d[16][('principal', 'component')]  =  34


#############################facendo queste modifiche al dizionario d,

def forza_arco(freq1,freq2,valori):
    return min(freq1,freq2)/valori

marta = []
# DEFINIAMO I LEGAMI TRA OGNI NODO

tot = 0
for i in range(0,len(d)-1):  # SELEZIONA L'i-esimo DIZIONARIO DA ANALIZZARE
    bigrammi_tot = list(d[i].keys())
    valori_tot = list(d[i].values())

    for j in range(0,len(bigrammi_tot)-1):# VADO AD ANALIZZARE DENTRO IL DIZIONARIO, SELEZIONANDO IL J-ESIMO BIGRAMMA
        bigramma_1 = bigrammi_tot[j]

        for z in range(j+1,len(bigrammi_tot)): # LO CONFRONTO CON TUTTI GLI ALTRI BIGRAMMI NEL DIZIONARIO
            bigramma_2 = bigrammi_tot[z]
            tot = forza_arco(valori_tot[j],valori_tot[z],l_cap[i]) +tot

            inz = i + 1
            for h in range(inz,len(d)): # ENTRO NEL H-ESIMO DIZIONARIO
                bigrammi_2 = list(d[h].keys())
                valori_2 = list(d[h].values())

                for k in range(0,len(bigrammi_2)): # ENTRO NEL K-ESIMO BIGRAMMA DEL DIZIONARIO H
                    if bigramma_1 == bigrammi_2[k]:
                        for t in range(0,len(bigrammi_2)):
                            if bigramma_2 == bigrammi_2[t]:
                                tot = forza_arco(valori_2[k],valori_2[t],l_cap[h]) + tot
            marta.append([bigramma_1,bigramma_2,tot])
            tot = 0

app = []
for i in range(0,len(marta)-1):
    for j in range(i+1,len(marta)):
        if marta[i][0] == marta[j][0] and marta[i][1] == marta[j][1]:
            app.append(j)

marta.remove(marta[357])
marta.remove(marta[139])
marta.remove(marta[119])
marta.remove(marta[118])


# FILE FOR GEPHY
import csv

with open("DatiGrezzi.csv","w", newline='') as f:
    thewriter = csv.writer(f)

    thewriter.writerow(['Source', 'Target', 'Weight'])

    for i in marta:
        thewriter.writerow(i)


###################### COSTRUZIONE DELLA MATRICE DI ADIACENZA
import pandas as pd
import numpy as np

data = pd.read_csv("OfficialEdges.csv",sep=';', header=None)
edges = data.drop([0], axis=0).to_numpy()

data = pd.read_csv("OfficialNodes.csv",sep=';', header=None)
nodi = data.drop([0], axis=0).to_numpy()

# MATRICE DI ADIACENZA
matrix = np.zeros((nodi.shape[0],nodi.shape[0]))
for i in range(0,edges.shape[0]):
    matrix[int(edges[i,0]),int(edges[i,1])] = float(edges[i,2])
    matrix[int(edges[i, 1]), int(edges[i, 0])] = float(edges[i, 2])

############################# INIZIAMO LO STUDIO DELLA RETE
import networkx as nx
import matplotlib.pyplot as plt

G = nx.from_numpy_matrix(matrix)
print(nx.info(G))

# WEIGHTED DEGREE
degree = []
alpha = 1 # Da far variare per 0.5, 1 e 1.5
for i in range(0, matrix.shape[0]):
    s = 0
    k = 0
    for j in range(0,matrix.shape[1]):
        if matrix[i,j] != 0:
            k = k + 1
            s = s + matrix[i,j]
    degree.append(k * (s/k) ** alpha)


plt.hist(degree)
plt.title("Istrogramma dei gradi")
plt.xlabel("Grado pesato")
plt.ylabel("Frequenza")
plt.show()


# BETWEENNESS CENTRALITY
from collections import defaultdict

alpha = 0.5 # da far variare, ma comunque è una nostra scelta

class Graph():
    def __init__(self):
        """
        self.edges is a dict of all possible next nodes
        e.g. {'X': ['A', 'B', 'C', 'E'], ...}
        self.weights has all the weights between two nodes,
        with the two nodes as a tuple as the key
        e.g. {('X', 'A'): 7, ('X', 'B'): 2, ...}
        """
        self.edges = defaultdict(list)
        self.weights = {}

    def add_edge(self, from_node, to_node, weight):
        # Note: assumes edges are bi-directional
        self.edges[from_node].append(to_node)
        self.edges[to_node].append(from_node)
        self.weights[(from_node, to_node)] =  int( 1 / ( float(weight) + 0.00000000001 )** alpha )
        self.weights[(to_node, from_node)] =  int( 1 / ( float(weight) + 0.00000000001 )** alpha )
graph = Graph()

edges[:,0] = edges[:,0].astype('str')
edges[:,1] = edges[:,1].astype('str')
edges1 = list(edges)

for edge in edges1:
    graph.add_edge(*edge)


def dijsktra(graph, initial, end):
    # shortest paths is a dict of nodes
    # whose value is a tuple of (previous node, weight)
    shortest_paths = {initial: (None, 0)}
    current_node = initial
    visited = set()

    while current_node != end:
        visited.add(current_node)
        destinations = graph.edges[current_node]
        weight_to_current_node = shortest_paths[current_node][1]

        for next_node in destinations:
            weight = graph.weights[(current_node, next_node)] + weight_to_current_node
            if next_node not in shortest_paths:
                shortest_paths[next_node] = (current_node, weight)
            else:
                current_shortest_weight = shortest_paths[next_node][1]
                if current_shortest_weight > weight:
                    shortest_paths[next_node] = (current_node, weight)

        next_destinations = {node: shortest_paths[node] for node in shortest_paths if node not in visited}
        if not next_destinations:
            return "Route Not Possible"
        # next node is the destination with the lowest weight
        current_node = min(next_destinations, key=lambda k: next_destinations[k][1])

    # Work back through destinations in shortest path
    path = []
    while current_node is not None:
        path.append(current_node)
        next_node = shortest_paths[current_node][0]
        current_node = next_node
    # Reverse path
    path = path[::-1]
    return path

dijsktra(graph, '0', '40') # ESEMPIO OUTPUT

d1 = []
for i in range(0,nodi.shape[0]):
    for j in range(0,nodi.shape[0]):
        if nodi[i,0] != nodi[j,0]:
            d1.append(dijsktra(graph, str(i), str(j)) )

bet = []
for i in range(0,nodi.shape[0]):
    count  = 0
    tot = 0
    for j in range(0,len(d1)):
        if d1[j] != 'Route Not Possible':
            tot = tot + 1
        if d1[j][0] != str(nodi[i,0]) and d1[len(d1[j])] != str(nodi[i,0]):
            for k in range(0,len(d1[j])):
                if d1[j][k] == str(nodi[i,0]):
                    count = count + 1
                    break
    bet.append(count/tot)

plt.hist(bet)
plt.title("Istrogramma della betweenness")
plt.xlabel("Betweenness")
plt.ylabel("Frequenza")
plt.show()

# Densità grafo
density = nx.density(G)

####### Essendo un grafico disconnesso, andiamo a calcolarci the average shortest path length PER LA COMPONENTE PIU GRANDE
largest_cc = list(max(nx.connected_components(G), key=len))

index = []
for i in range(0,edges.shape[0]):
    count = 0
    for j in range(0,len(largest_cc)):
        if int(edges[i,0]) != largest_cc[j]:
            count = count + 1
    if count == len(largest_cc):
        index.append(i)
for i in range(0,edges.shape[0]):
    count = 0
    for j in range(0,len(largest_cc)):
        if int(edges[i,1]) != largest_cc[j]:
            count = count + 1
    if count == len(largest_cc):
        index.append(i)

from collections import OrderedDict
unique = list(OrderedDict.fromkeys(index))

unique = sorted(unique,reverse=True)
edges_componente_maggiore = edges
for i in range(0,len(unique)):
    edges_componente_maggiore = np.delete(edges_componente_maggiore, unique[i], 0)
edges_componente_maggiore.shape


# MATRICE DI ADIACENZA COMPONENTE MAGGIORE
matrix_cc = np.zeros((nodi.shape[0],nodi.shape[0]))
for i in range(0,edges_componente_maggiore.shape[0]):
    matrix_cc[int(edges_componente_maggiore[i,0]),int(edges_componente_maggiore[i,1])] = float(edges_componente_maggiore[i,2])
    matrix_cc[int(edges_componente_maggiore[i,1]),int(edges_componente_maggiore[i,0])] = float(edges_componente_maggiore[i,2])

delet = [97,96,50,49,48,47,46,45,44]

for i in range(0,len(delet)):
    matrix_cc = np.delete(matrix_cc, delet[i], 0)
    matrix_cc = np.delete(matrix_cc, delet[i], 1)
matrix_cc.shape

ASPL = nx.average_shortest_path_length(G_cc) # 3.008
nx.diameter(G_cc)  #Lunghezza del più lungo shortest path della componente più grande

################# COMMUNITY DETECTION
pip install python-louvain
import community

# Algoritmo di Girvan_Newman
communities = next(nx.community.girvan_newman(G))
len(communities)  # vengono identificate 4 comunità
print(communities)  # tupla in cui ci sono le liste dei nodi facenti parte dic iascuna comunità

# creo il dizonario a partire dall'oggetto communities
# prendo il numero di nodo e lo associo al progressivo della community

part_girvan_newman = dict()
id_comm = 0

for comm in communities:
    for node in comm:
        part_girvan_newman.setdefault(node, id_comm)  # stedefault associa chiave e valore
    id_comm += 1

community.modularity(part_girvan_newman, G)

# Algoritmo di Louvain
dc = community.best_partition(G)  # restituisce un dizionario in cui ciascun nodo è associato alla comuntà
dc
# 8 comunità
community.modularity(dc, G)


################## CODICE CHE  SERVE PER IMPORTARE SU GEPHI
import csv

louvain = list(dc.values())
with open("Louvain.csv","w", newline='') as f:
    thewriter = csv.writer(f)

    thewriter.writerow(['Category'])
    thewriter.writerows([[hit] for hit in louvain])

################################################COSE CHE ABBIAMO DECISO DI NON USARE:
    #STEMMING
    #TOGLIERE PAROLE RARE E FREQUENTI TRAMITE DIZIONARIO
    #TEST D'INDIPENDENZA

# STEMMING:
# Stemming is a process of linguistic normalization, which reduces words to their word root word or chops off the derivational affixes.
# For example, connection, connected, connecting word reduce to a common word "connect".

# from nltk.stem import PorterStemmer
#ps = PorterStemmer()
#stemmed_words=[]
#for w in finali:
 #   stemmed_words.append(ps.stem(w))


# TOGLIERE PAROLE TROPPO FREQUENTI O TROPPO RARE TRAMITE L'UTILIZZO DI UN DIZIONARIO
#dict1=Counter(finali)
#vv={k: v for k, v in sorted(dict1.items(), key=lambda item: item[1])}
#dict2=dict((k, v) for k, v in dict1.items() if v <= 100)#potrebbe essere interessante togliere anche le parole che si ripetono troppo
#dict2

# Rivediamo graficamente ora i risultati dopo la pulizia
#import matplotlib.pyplot as plt
# fdist = FreqDist(dict2) #si puo fare sia su una stringa che su un dizionario
# fdist.plot(30,cumulative=False)
# plt.show()
# print ("fatto")


#fdist = FreqDist(bigrams)
#plt.figure(figsize=(20, 10))
#fdist.plot(100,cumulative=False)
#plt.savefig( 'myfig.jpg' ) # Comando che salva l'immagine nella cartella TextMining

#### TEST D'INDIPENDENZA
# Raw popularity count is too crude of a measure.
# We have to find more clever statistics to be able to pick out meaningful phrases easily.
# For a given pair of words, the method tests two hypotheses on the observed dataset.
# Hypothesis 1 (the null hypothesis) says that word 1 appears independently from word 2.
# Hypothesis 2 (the alternate hypothesis) says that seeing word 1 changes the likelihood of seeing word 2.

# Hence, the likelihood ratio test for phrase detection (a.k.a. collocation extraction) asks the following question:
# Are the observed word occurrences in a given text corpus more likely to have been generated from a model where
# the two words occur independently from one another?

#import math

#def logL(p,k,n):
#    return k * math.log(p) + (n-k)* math.log( 1 - p + 0.00000000001)

# from collections import OrderedDict
# unici = list(OrderedDict.fromkeys(bigrams))
# unici

# def log_likelihood_statistic (p,k,n):
#     p1 = p[0]
#     p2 = p[1]
#     p = p[2]
#     k1 = k[0]
#     k2 = k[1]
#     n1 = n[0]
#     n2 = n[1]

#     return 2 * ( logL(p1,k1,n1) + logL(p2,k2,n2) - logL(p,k1,n1) -logL(p,k2,n2) )


# p = np.zeros(3)
# ris = np.zeros(len(unici))

# # Vado a calcolare la log-rapporto di verosimiglianza!!
# for i in range(0,len(unici)):
#     k = np.zeros(2)
#     n = np.zeros(2)
#     for h in range(0,len(bigrams)):

#             if unici[i][0] == bigrams[h][0]:
#                 k[0] = k[0] + 1
#                 n[0] = n[0] + 1
#             if unici[i][0] == bigrams[h][1]:
#                 n[0] = n[0] + 1

#             if unici[i][1] == bigrams[h][1]:
#                 k[1] = k[1] + 1
#                 n[1] = n[1] + 1
#             if unici[i][1] == bigrams[h][0]:
#                 n[1] = n[1] + 1

#     p[0] = (k[0] / n[0])
#     p[1] = (k[1] / n[1])
#     p[2] = (k[0] + k[1]) / (n[0] + n[1])

#     ris[i] = log_likelihood_statistic(p,k,n)

# ris = list(ris)
# log_ordinata= sorted(range(len(ris)), key=lambda k: ris[k])  #OCCHIO AL MENO !



# for k in range(0,30):
#   print(unici[log_ordinata[k]])


