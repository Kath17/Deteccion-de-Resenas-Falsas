from nltk.tokenize import sent_tokenize, word_tokenize 
from sklearn import model_selection, preprocessing
import warnings 
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Conv1D, MaxPooling1D
import keras
from keras import backend as K
from keras.layers.embeddings import Embedding
from keras.layers import Dense, Activation
from keras.callbacks import LambdaCallback

from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from keras.layers.convolutional import Conv2D

warnings.filterwarnings(action = 'ignore') 
  
import gensim 
from gensim.models import Word2Vec 
  

# ----------------------------------------------------------------------------------#
# ----------------------------- Prepare DB for word2vec model ----------------------#
# ----------------------------------------------------------------------------------#

#Reads preprocesed yelp data
#df = pd.read_csv('Yelp_prepro.txt', sep='\t',usecols=[0,1],names=["review","label"])
df = pd.read_csv('New_yelp_sample.txt', sep='\t',usecols=[0,1],names=["review","label"])
#print("\nTamaño - Numero de oraciones:",df.shape)
df = df.dropna(thresh=1)

print(df.head(6))

# Create the list of list format of the custom corpus for gensim modeling 
# ------------------------------ Data -------------------------#
# ----- data = [["i", "am", "cat"], ["hello", "how"]]

print("\n Creamos una lista de listas para ingresar los datos para poder pasarlo por el Word2Vec")
data = []
max_len = 0
for row in df['review']:
  #print(row)
  #print(str(row).split(" "))
  oracion = str(row).split(" ")
  if(max_len < len(oracion)):
    max_len = len(oracion)
  data.append(str(row).split(" "))

print("\nLongitud de oracion más grande: ",max_len)     # 2730    # 1000  --- #new sample:
#max_len = 1000
print("\nNumero de oraciones", len(data))         # 608457     # ---- new sample: 
print(data[:5])

# ---------------------------- Labels --------------------------# 
#   -------------------------- 1 or -1 ----------------------   #

print("\n Guardamos las etiquetas en una lista aparte:")
labels = df['label']
print("\nLabels:\n",labels[:5])


# -------------------------------- Glove model ------------------------- #
# ----------------------------------------------------------------------#
"""
embeddings_index = dict()
f = open('glove.6B/glove.6B.100d.txt')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()
"""

# ------------------------------ Create Word2Vec model -------------------------------#
#   --------------------------------- save file --------------------------------------#
print("\n  --------- Creando modelo Word2Vec --------")
#model = gensim.models.Word2Vec(data, min_count = 1, size = 100, window = 10, iter=10) 
#model.save("word2vec.model")

model = Word2Vec.load("word2vec.model")
#model = Word2Vec.load("word2vec_total.model")

word_vectors = model.wv
pretrained_weights = model.wv.syn0
vocab_size, embedding_size = pretrained_weights.shape


print('\nLongitud del vocabulario: {}'.format(len(word_vectors.vocab)))   #436395    #23770 #--- new sample:
#print('vocab_size',vocab_size)                             #436395     #23770  #----- new sample: 
print('\nTamaño de cada palabra incrustada (embedding size): ', embedding_size)                             #100

print("\nModelo de la palabra coffe: \n", model['coffe'])
#print("model - serv: ", model['serv'])

print("\n Probando el modelo resultante:")
print("\nSimilitud entre: coffe y capuccino\n", model.similarity('coffe', 'cappuccino'))
print("\nSimilitud entre: coffe y relax\n", model.similarity('coffe', 'relax'))

def word2idx(word):
  try:
    return model.wv.vocab[word].index
  except:
    return 0
def idx2word(idx):
  return model.wv.index2word[idx]

#print("model.wv.vocab[coffe]: ", model.wv.vocab["coffe"])
#print("word2idx --- drink: ", word2idx("drink"))
#print("idx2word --- 0 : ", idx2word(0))


"""

# Guardamos los vectorel del modelo Word2Vec a una nueva matriz
embedding_matrix = np.zeros((len(model.wv.vocab)+1, 100))
for i, vec in enumerate(model.wv.vectors):
  embedding_matrix[i] = vec

print("\nEmbedding_matrix: ---> ",embedding_matrix[:3])

# how many features should the tokenizer extract
features = 500
tokenizer = Tokenizer(num_words = features)
# fit the tokenizer on our text
tokenizer.fit_on_texts(text)

"""


"""
"""
# ---------------------------------------------------------------------#
# ------------------------ Preparing data for LSTM --------------------//
# ---------------------------------------------------------------------#


print('\nPreparing the data for LSTM...')
train_x = np.zeros([len(data), max_len], dtype=np.int32)
train_y = np.zeros([len(data)], dtype=np.int32)
for i,sentence in enumerate(data):
  #for t, word in enumerate(sentence[:-1]):
  for t, word in enumerate(sentence):
    train_x[i,t] = word2idx(word) 
  train_y[i] = word2idx(sentence[-1])
  #train_y[i] = word2idx()
print('train_x shape:', train_x.shape)
#print('train_y shape:', train_y.shape)

print(" train_x: ",train_x[:5])
#print(" train_y: ",train_y[:5])



print('\nTraining LSTM...')

# --------------------------------------------------------------------#
# -------------------- Adding embedding Layer ------------------------#
# --------------------------------------------------------------------#

lstm_model = Sequential()
lstm_model.add(Embedding(input_dim=vocab_size, output_dim=embedding_size, weights=[pretrained_weights]))#, trainable=False))

# --------------------------------------------------------------------#
# --------------------------- Adding CNN -----------------------------#
# --------------------------------------------------------------------#

#lstm_model.add(Conv1D(4, 2, activation='relu', padding='valid'))
lstm_model.add(Conv1D(4, 2, activation='sigmoid', padding='valid'))
lstm_model.add(MaxPooling1D(pool_size=4))

# --------------------------------------------------------------------#
# --------------------------- Adding LSTM ----------------------------#
# --------------------------------------------------------------------#

lstm_model.add(LSTM(units=embedding_size))
lstm_model.add(Dense(1, activation='sigmoid'))
#lstm_model.add(Activation('softmax'))
lstm_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#lstm_model.fit(train_x, train_y, batch_size=128, epochs=20, callbacks=[LambdaCallback(on_epoch_end=on_epoch_end)])
#-----------------despues -------------#

lstm_model.fit(train_x, np.array(labels), validation_split=0.4,epochs=3)

print("Termino")


"""
# ----------------------------- Applying LSTM classification --------------------------#
# split the dataset into training and validation datasets 
train_x, valid_x, train_y, valid_y = model_selection.train_test_split(df['review'], df['label'])
print("Train en x:",train_x.shape)
print("Valid en x:",valid_x.shape)
print("Train en y:",train_y.shape)
print("Valid en y:",valid_y.shape)

# label encode the target variable 
#encoder = preprocessing.LabelEncoder()
#train_y = encoder.fit_transform(train_y)
#valid_y = encoder.fit_transform(valid_y)

#model.add(LSTM(units=50, return_sequences=True, input_shape=(features_set.shape[1], 1)))

"""



