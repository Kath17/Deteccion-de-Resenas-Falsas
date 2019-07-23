import pandas as pd
import csv
import string
import nltk
from nltk.corpus import stopwords
import re
from nltk.stem.snowball import SnowballStemmer

# 1. ========================= Crear un nuevo file con los datos necesarios ============================

# df = pd.read_csv('reviewContent.txt', sep='\t',usecols=[0,1,3],names=["ID","ID2","review"])#, usecols=[0,3])
# df2 = pd.read_csv('metadata.txt', sep='\t', usecols=['ID','ID2','labe'])
# print("size:",df.shape)
# print("size:",df2.shape)
# merged = df.merge(df2, how='left', on=['ID','ID2'])
#
# print(df.head(10))
# print(df2.head(10))
# print(merged.head(30))
#
# merged.to_csv('Yelp.txt', sep='\t',header=None, mode='a',columns=['review','labe'],index=False)
# print("size2:",merged.shape)

# ------- Reading new file
# df = pd.read_csv('Yelp.txt', sep ='\t',header=None)
# print(df.head(20))

# ------- Sacar duplicados -- Not used
# merged.drop_duplicates(subset ='ID', keep = 'first', inplace = True)

# 2. ============================ Remove numeric and empty text =====================================

# ------- Reading new file
rm_quote = lambda x: x.replace('"', '')
df = pd.read_csv('Yelp.txt', sep ='\t',names=["review","label"],converters={'\"review\"': rm_quote})
print(df.head(20))
print(df.shape)

# ------- Eliminando
df= df.dropna()  # Eliminar comentarios nulos
df = df[df.review.apply(lambda x: False == x.isnumeric())]  # Eliminar comentarios de números
df = df[df.review.apply(lambda x: x !="")]        # Eliminar comentarios vacios

print(df.shape)
print(df.head(20))

# 3. ============================ Lowercase, Stop words, etc =====================================

def limpiar(comentario):

    comentario = comentario.translate(string.punctuation) # Remover puntuación
    comentario = comentario.lower().split() # Convertir a lowercase

    # Remover stop words
    stops = set(stopwords.words("english"))
    comentario = [c for c in comentario if not c in stops and len(c) >= 3]

    # --- LIMPIANDO LAS RESEÑAS --------#
    comentario = " ".join(comentario)    ## Clean the comentario
    comentario = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", comentario)
    comentario = re.sub(r"what's", "what is ", comentario)
    comentario = re.sub(r"\'s", " ", comentario)
    comentario = re.sub(r"\'ve", " have ", comentario)
    comentario = re.sub(r"n't", " not ", comentario)
    comentario = re.sub(r"i'm", "i am ", comentario)
    comentario = re.sub(r"\'re", " are ", comentario)
    comentario = re.sub(r"\'d", " would ", comentario)
    comentario = re.sub(r"\'ll", " will ", comentario)
    comentario = re.sub(r",", " ", comentario)
    comentario = re.sub(r"\.", " ", comentario)
    comentario = re.sub(r"!", " ! ", comentario)
    comentario = re.sub(r"\/", " ", comentario)
    comentario = re.sub(r"\^", " ^ ", comentario)
    comentario = re.sub(r"\+", " + ", comentario)
    comentario = re.sub(r"\-", " - ", comentario)
    comentario = re.sub(r"\=", " = ", comentario)
    comentario = re.sub(r"'", " ", comentario)
    comentario = re.sub(r"(\d+)(k)", r"\g<1>000", comentario)
    comentario = re.sub(r":", " : ", comentario)
    comentario = re.sub(r" e g ", " eg ", comentario)
    comentario = re.sub(r" b g ", " bg ", comentario)
    comentario = re.sub(r" u s ", " american ", comentario)
    comentario = re.sub(r"\0s", "0", comentario)
    comentario = re.sub(r" 9 11 ", "911", comentario)
    comentario = re.sub(r"e - mail", "email", comentario)
    comentario = re.sub(r"\s{2,}", " ", comentario)    ## Stemming

    comentario = comentario.split()   #Steamming de palabras
    stemmer = SnowballStemmer('english')
    stemmed_words = [stemmer.stem(word) for word in comentario]
    comentario = " ".join(stemmed_words)
    # ---------------------------#

    comentario = " ".join(comentario)
    return comentario

df['review'] = df['review'].map(lambda x: limpiar(x))
print(df.head(20))

df.to_csv('Yelp_pre.txt', sep='\t',header=None, mode='a',columns=['review','label'],index=False)
print("size preprocesado:",df.shape)
