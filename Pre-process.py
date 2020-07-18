import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from IPython import display
%matplotlib inline

#Get the vocabulary of your dataset
def Get_vocab(l):
  '''
  Get the vocabulary of your dataset, obtain all the different letters in your dataset
  l = (pandas.core.frame.DataFrame) sentences from your dataset l = ["Hola","Como estas?", "Bien y tu?", "Hola"]
  
  Return:
  vocab: (list) vocabulary of your dataset
  array_joined: (str) your data in one single string separated by \n\n
  '''
  l = []
  for i in range(len(data.columns)):
    l.extend(list(data.iloc[i,:]))
  array_joined = "\n\n".join(l)
  vocab = sorted(set(array_joined)) 
  return vocab, array_joined


#Get the meaning of an array of numbers
def Idx2char(vocab,l):
  '''
  Function to get again the meaning of the numbers (convert number to strings)
  vocab = (list) list of all different letters used in the dataset
  l = (numpy.ndarray)

  Return
  String_generated: (str) numbers converted in letters in a single string
  '''
  String_generated = ""
  idx2char= np.array(vocab)
  try:
    l = l.flatten()
  except:
    l = tf.keras.backend.flatten(l)
  for i in l:
    String_generated += idx2char[int(i)]
  return String_generated


#Transform the data in a vectorized string (Transform the data in numbers)
def vectorize_string(vocab, sentence):
  '''
  Vectorize your string
  sentence = (list) string to vectorize
  vocab = (list) list of all different letters used in the dataset

  Return:
  vector = (numpy.ndarray) the vectorized string (your letters transformed in numbers)
  char2idx = (dict) map of the letters in numbers
  '''
  char2idx = {u:i for i, u in enumerate(vocab)}
  vector = np.zeros((1,len(string)))
  for i,j in enumerate(string):
    vector[0,i] = char2idx[j]
  return vector, char2idx

#Get a batch of words vectorized, so if you have an input of "Hell", the output will be "ello" (length + 1)
def get_batch(vectorized_strings, seq_length, batch_size):
  '''
  Function to get examples of the dataset
  vectorized_strings: (numpy.ndarray) array with all vectorized strings (numbers representing letters)
  seq_length: (int) number of letters you want in that pair of samples (Hell-ello would be seq_length = 4)
  batch_size: (int) how many pairs of samples you want it? (Hell-ello; Ho-ow;, mornin-orning would be batch_size = 3)

  Return:
  x_batch: (numpy.ndarray) Batch for the input
  y_batch: (numpy.ndarray) Batch for the output
  '''
  # the length of the vectorized songs string
  n = vectorized_strings.shape[1] - 1
  # randomly choose the starting indices for the examples in the training batch
  idx = np.random.choice(n-seq_length, batch_size)

  input_batch = [vectorized_strings[0,j:j+seq_length] for j in idx]
  output_batch = [vectorized_strings[0,j+1:j+seq_length+1] for j in idx]
  

  # x_batch, y_batch provide the true inputs and targets for network training
  x_batch = np.reshape(input_batch, [batch_size, seq_length])
  y_batch = np.reshape(output_batch, [batch_size, seq_length])
  return x_batch, y_batch
