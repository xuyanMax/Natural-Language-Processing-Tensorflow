# WEEK-2

## Summary

## Week Mission
Laurence Moroney
> Last week you saw how to use the Tokenizer to prepare your text to be used by a neural network by converting words into numeric tokens, and sequencing sentences from these tokens. This week you'll learn about `Embeddings`, where these tokens are `mapped as vectors in a high dimension space`. With Embeddings and labelled examples, these vectors can then be tuned so that words with similar meaning will have a similar direction in the vector space. This will begin the process of `training a neural network to understand sentiment in text` -- and you'll begin by looking at movie reviews, training a neural network on texts that are labelled 'positive' or 'negative' and determining which words in a sentence drive those meanings.
> 
## IMDB Datasets

Please find the link to he IMDB reviews dataset [here](http://ai.stanford.edu/~amaas/data/sentiment/)

You will find here 50,000 movie reviews which are classified as positive of negative.


### Load imdb reviews data and split training and validation datasets
```python
import tensorflow as tf
import numpy as np

import tensorflow_datasets as tfds
imdb, info = tfds.load("imdb_reviews", with_info=True, as_supervised=True)

train_data, test_data = imdb['train'], imdb['test']
# 25000
training_sentences = []
training_labels = []
# 25000
testing_sentences = []
testing_labels = []

# str(s.tonumpy()) is needed in Python3 instead of just s.numpy()
for s,l in train_data:
  training_sentences.append(str(s.numpy()))
  training_labels.append(l.numpy())
  
for s,l in test_data:
  testing_sentences.append(str(s.numpy()))
  testing_labels.append(l.numpy())
  
training_labels_final = np.array(training_labels)
testing_labels_final = np.array(testing_labels)

```

### Create sequencing sentences from tokens
```python
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocess ing.sequence import pad_sequences

vocab_size = 10000
embedding_dim = 16
max_length = 120
trunc_type='post'
oov_tok = "<OOV>"

tokenizer = Tokenizer(num_words = vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(training_sentences)
word_index = tokenizer.word_index

# So each of my sentences is now a list of numbers.
sequences = tokenizer.texts_to_sequences(training_sentences)
# padding sequences ensure same length: either padded out or truncated to suit
padded = pad_sequences(sequences,maxlen=max_length, truncating=trunc_type)

testing_sequences = tokenizer.texts_to_sequences(testing_sentences)
testing_padded = pad_sequences(testing_sequences,maxlen=max_length)
```
### Train with Sequential Neural Network 

#### Take reverse world index
```python
# taken my reverse word index, and I can decode my review by taking a look at # the numbers in that review and reversing that into a word.
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])

print(decode_review(padded[1]))
print(training_sentences[1])
```

#### Embedding 
**Quote from Laurence Moroney**

>The full scope of how embeddings work is beyond the scope of this course. But think of it like this. You have words in a sentence and often words that have similar meanings are close to each other. So in a movie review, it might say that the movie was dull and boring, or it might say that it was fun and exciting. 
>
> `So what if you could pick a vector in a higher-dimensional space say 16 dimensions`, and words that are found together are given similar vectors. Then over time, `words can begin to cluster together`. The meaning of the words can come from the labeling of the dataset. So in this case, we say a negative review and the words dull and boring show up a lot in the negative review so that they have similar sentiments, and they are close to each other in the sentence. Thus their vectors will be similar. As the neural network trains, it can then learn these vectors associating them with the labels to come up with what's called an `embedding` i.e., the vectors for each word with their associated sentiment. The results of the embedding will be a `2D array` with the length of the sentence and the embedding dimension for example 16 as its size. So we need to `flatten it out` in much the same way as we needed to flatten out our images. We then feed that into a dense neural network to do the classification. Often in natural language processing, a different layer type than a flatten is used, and this is a `global average pooling 1D`. The reason for this is the size of the output vector being fed into the dense. So for example, if I show the summary of the model with the flatten that we just saw, it will look like this. Or alternatively, you can use a Global Average Pooling 1D like this, which averages across the vector to flatten it out. Your model summary should look like this, which is simpler and should be a little faster. Try it for yourself in colab and check the results. Over 10 epochs with global average pooling, I got an accuracy of 0.9664 on training and 0.8187 on test, taking about 6.2 seconds per epoch. With flatten, my accuracy was 1.0 and my validation about 0.83 taking about 6.5 seconds per epoch. So it was a little slower, but a bit more accurate. Try them both out, and experiment where the results for yourself.

```python
e = model.layers[0]
weights = e.get_weights()[0]
print(weights.shape) # shape: (vocab_size, embedding_dim)

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(tokenizer.vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.Flatten(), # or use tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(6, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()

# Model: "sequential"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #   
# =================================================================
# embedding (Embedding)        (None, 120, 16)           160000    
# _________________________________________________________________
# flatten (Flatten)            (None, 1920)              0         
# _________________________________________________________________
# dense (Dense)                (None, 6)                 11526     
# _________________________________________________________________
# dense_1 (Dense)              (None, 1)                 7         
# =================================================================
# Total params: 171,533
# Trainable params: 171,533
# Non-trainable params: 0
# ___________________________
```

### Output Embedding Metadata
Quote from Laurence

> So now let's take a look at what we'll do to view this in the embedding projector. So first of all, I'm going to take the output of my embedding, which was modeled out `layer zero`, and we can see that there were `10,000 possible words and I had 16 dimensions`. 
> 
> Here is where I'm going to iterate through that array to pull out the 16 dimensions, the values for the 16 dimensions per word and write that as out_V, which is my vectors.tsv. 
> 
> Then the actual word associated with that will be written to out_M, which is my meta.tsv. So if I run that, it we'll do its trick and if you're running in Colab this piece of code, will then allow me to just download those files. So it'll take a moment and they'll get downloaded. There they are, vecs.tsv and meta.tsv. So if I now come over to the embedding projector, we see its showing right now the `Word2Vec 10K`. So if I scroll down here and say `load data`, I'll choose file, I'll take the vecs.tsv. I'll choose file. I'll take the meta.tsv, then load. I click outside and now I see this. But if I `spherize` the data, you can see it's clustered like this. We do need to improve it a little bit but we can begin to see that the `words have been clustered in both the positive and negative`. So for example if I search for the word boring, we can see like the nearest neighbors for boring are things like stink or unlikeable, prom, unrealistic wooden, devoid, unwatchable, and proverbial. So if come over here we can see. These are bad words. These are words showing a negative looking review.
 
```python
import io

out_v = io.open('vecs.tsv', 'w', encoding='utf-8')
out_m = io.open('meta.tsv', 'w', encoding='utf-8')
for word_num in range(1, vocab_size):
  word = reverse_word_index[word_num]
  out_m.write(word + "\n")
  embeddings = weights[word_num]
  out_v.write('\t'.join([str(x) for x in embeddings]) + "\n")
out_v.close()
out_m.close()
```

### Tensorflow Projecter 
[Visualize Embedding](https://projector.tensorflow.org)

### Apply model on validation sets
```python
num_epochs = 10
model.fit(padded, training_labels_final, epochs=num_epochs, validation_data=(testing_padded, testing_labels_final))


# Train on 25000 samples, validate on 25000 samples
# W0818 16:52:47.867466 140151655249792 deprecation.py:323] From /usr/local/lib/python3.6/dist-packages/tensorflow/python/ops/math_grad.py:1250: add_dispatch_support.<locals>.wrapper (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
# Instructions for updating:
# Use tf.where in 2.0, which has the same broadcast rule as np.where
# Epoch 1/10
# 25000/25000 [==============================] - 7s 265us/sample - loss: 0.4924 - acc: 0.7470 - val_loss: 0.3469 - val_acc: 0.8466
# Epoch 2/10
# 25000/25000 [==============================] - 5s 208us/sample - loss: 0.2520 - acc: 0.9000 - val_loss: 0.3575 - val_acc: 0.8471
# Epoch 3/10
# 25000/25000 [==============================] - 5s 208us/sample - loss: 0.1257 - acc: 0.9622 - val_loss: 0.4358 - val_acc: 0.8276
# Epoch 4/10
# 25000/25000 [==============================] - 5s 206us/sample - loss: 0.0370 - acc: 0.9941 - val_loss: 0.5166 - val_acc: 0.8275
# Epoch 5/10
# 25000/25000 [==============================] - 5s 207us/sample - loss: 0.0096 - acc: 0.9990 - val_loss: 0.5792 - val_acc: 0.8275
# Epoch 6/10
# 25000/25000 [==============================] - 5s 206us/sample - loss: 0.0024 - acc: 1.0000 - val_loss: 0.6370 - val_acc: 0.8276
# Epoch 7/10
# 25000/25000 [==============================] - 5s 208us/sample - loss: 0.0011 - acc: 1.0000 - val_loss: 0.6824 - val_acc: 0.8295
# Epoch 8/10
# 25000/25000 [==============================] - 5s 210us/sample - loss: 5.6307e-04 - acc: 1.0000 - val_loss: 0.7246 - val_acc: 0.8298
# Epoch 9/10
# 25000/25000 [==============================] - 5s 208us/sample - loss: 3.2410e-04 - acc: 1.0000 - val_loss: 0.7635 - val_acc: 0.8298
# Epoch 10/10
# 25000/25000 [==============================] - 5s 208us/sample - loss: 1.9135e-04 - acc: 1.0000 - val_loss: 0.8007 - val_acc: 0.8294
# <tensorflow.python.keras.callbacks.History at 0x7f773c090748>
```

## Loss Function

Think about loss in this context, as a `confidence` in the prediction. So while the number of accurate predictions increased over time, what was interesting was that the confidence per prediction effectively decreased. You may find this happening a lot with text data. 

Ways to handle such cases:
- One way to do this is to explore the differences as you tweak the hyperparameters.
    + So for example, if you consider these changes, a decrease in vocabulary size(10,000 -> 1,000), and taking shorter sentences, reducing the likelihood of padding
    + Another tweak. Changing the number of dimensions using the embedding was also tried

## Pre-tokenized Datasets & Subwords text encoder 

 > Well, the keys in the fact that we're using sub-words and not for-words, sub-word meanings are often nonsensical and it's only when we put them together in sequences that they have meaningful semantics. Thus, some way from learning from sequences would be a great way forward, and that's exactly what you're going to do next week with recurrent neural networ

```python
# NOTE: PLEASE MAKE SURE YOU ARE RUNNING THIS IN A PYTHON3 ENVIRONMENT

import tensorflow as tf
print(tf.__version__)

# Uncomment and run this if you don't have TensorFlow 2.0x [Check for latest 2.0 instructions at https://www.tensorflow.org/versions/r2.0/api_docs/python/tf]
#!pip install tensorflow==2.0.0-beta0

# Double check TF 2.0x is installed. If you ran the above block, there was a 
# 'reset all runtimes' button at the bottom that you needed to press
import tensorflow as tf
print(tf.__version__)

# If the import fails, run this
# !pip install -q tensorflow-datasets

import tensorflow_datasets as tfds
imdb, info = tfds.load("imdb_reviews/subwords8k", with_info=True, as_supervised=True)

train_data, test_data = imdb['train'], imdb['test']

tokenizer = info.features['text'].encoder

print(tokenizer.subwords)

sample_string = 'TensorFlow, from basics to mastery'

tokenized_string = tokenizer.encode(sample_string)
print ('Tokenized string is {}'.format(tokenized_string))

original_string = tokenizer.decode(tokenized_string)
print ('The original string: {}'.format(original_string))

for ts in tokenized_string:
  print ('{} ----> {}'.format(ts, tokenizer.decode([ts])))

# One thing to take into account though, is the shape of the vectors coming from the tokenizer through the embedding, and it's not easily flattened. So we'll use Global Average Pooling 1D instead. Trying to flatten them, will cause a TensorFlow crash.
embedding_dim = 64
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(tokenizer.vocab_size, embedding_dim),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(6, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.summary()

num_epochs = 10

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

history = model.fit(train_data, epochs=num_epochs, validation_data=test_data)

import matplotlib.pyplot as plt


def plot_graphs(history, string):
  plt.plot(history.history[string])
  plt.plot(history.history['val_'+string])
  plt.xlabel("Epochs")
  plt.ylabel(string)
  plt.legend([string, 'val_'+string])
  plt.show()
  
plot_graphs(history, "accuracy")
plot_graphs(history, "loss")
In [0]:
e = model.layers[0]
weights = e.get_weights()[0]
print(weights.shape) # shape: (vocab_size, embedding_dim)

import io

out_v = io.open('vecs.tsv', 'w', encoding='utf-8')
out_m = io.open('meta.tsv', 'w', encoding='utf-8')
for word_num in range(1, tokenizer.vocab_size):
  word = tokenizer.decode([word_num])
  embeddings = weights[word_num]
  out_m.write(word + "\n")
  out_v.write('\t'.join([str(x) for x in embeddings]) + "\n")
out_v.close()
out_m.close()


try:
  from google.colab import files
except ImportError:
  pass
else:
  files.download('vecs.tsv')
  files.download('meta.tsv')
```


