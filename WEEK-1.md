
# Tokenize the text

Laurence Moroney

> The first step in understanding sentiment in text, and in particular when training a neural network to do so is the tokenization of that text. This is the process of converting the text into numeric values, with a number representing a word or a character. This week you'll learn about the Tokenizer and pad_sequences APIs in TensorFlow and how they can be used to prepare and encode text and sentences to get them ready for training neural networks!

```python
from tensorflow.keras.preprocessing.text import Tokenizer

sentences=[
        'I love my dog',
        'I love my cat!'
]
# create an instance of tokenizer, with a passive parameter num_words=100, way # too big in this case, since there are only 5 distinct word in sentences

# What tokenizer will do is take the top/most common 100 words by volume and 
# just encode those

# The tokenizer provides a word index property which returns a dictionary 
# containing [key,value] pairs, where the key is the word, and the value is 
# the token for that word

# <OOV> The idea here is that I'm going to create a new token, a special token # that I'm going to use for words that aren't recognized, that aren't in the 
# word index itself.
tokenizer = Tokenizer(num_words=100, oov_token="<OOV>")
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index
print(word_index) #{'i':1, 'my':3, 'dog':4, 'cat':5, 'love':2}
```
## Encoding texts to <word, token> pairs
 
```python
# It will go through the entire body of text and it will create a dictionary 
# with the key being the word and the value being the token for that word.
tokenizer.fit_on_texts(sentences)
```

## Convert sentences to sequences of tokens, with padding

```python
from tensorflow.keras.preprocessing.sequence import pad_sequences

# convert the words in those sentences to sequences of tokens(numbers)
sequences = tokenizer.texts_to_sequences(sentences)

# padding after the sentence and not before 
# truncate the sentece after exceeding the max length 
padded = pad_sequences(sequences, padding='post', maxlen=5, truncating='post')
```

## Practice Sarcasm Datasets

[Sarcasm in News Headlines Dataset by Rishabh Misra](https://rishabhmisra.github.io/publications/)
```python
!wget --no-check-certificate \
    https://storage.googleapis.com/laurencemoroney-blog.appspot.com/sarcasm.json \
    -O /tmp/sarcasm.json
  
import json

with open("/tmp/sarcasm.json", 'r') as f:
    datastore = json.load(f)


sentences = [] 
labels = []
urls = []
for item in datastore:
    sentences.append(item['headline'])
    labels.append(item['is_sarcastic'])
    urls.append(item['article_link'])

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
tokenizer = Tokenizer(oov_token="<OOV>")
tokenizer.fit_on_texts(sentences)

word_index = tokenizer.word_index
print(len(word_index))
print(word_index)
sequences = tokenizer.texts_to_sequences(sentences)
padded = pad_sequences(sequences, padding='post')
print(padded[0])
print(padded.shape)


```

console output 

```
--2019-08-15 16:51:56--  https://storage.googleapis.com/laurencemoroney-blog.appspot.com/sarcasm.json
Resolving storage.googleapis.com (storage.googleapis.com)... 74.125.20.128, 2607:f8b0:400e:c09::80
Connecting to storage.googleapis.com (storage.googleapis.com)|74.125.20.128|:443... connected.
HTTP request sent, awaiting response... 200 OK
Length: 5643545 (5.4M) [application/json]
Saving to: ‘/tmp/sarcasm.json’

/tmp/sarcasm.json   100%[===================>]   5.38M  --.-KB/s    in 0.02s   

2019-08-15 16:51:56 (232 MB/s) - ‘/tmp/sarcasm.json’ saved [5643545/5643545]

29657
{'<OOV>': 1, 'to': 2, 'of': 3, 'the': 4, 'in': 5, 'for': 6, 'a': 7, 'on': 8, 'and': 9, 'with': 10, 'is': 11, 'new': 12, 'trump': 13, 'man': 14,...}

[  308 15115   679  3337  2298    48   382  2576 15116     6  2577  8434
     0     0     0     0     0     0     0     0     0     0     0     0
     0     0     0     0     0     0     0     0     0     0     0     0
     0     0     0     0]
(26709, 40)
```

## IMDB Datasets

## Tensorflow Datasets as tfds

## Explore BBC News Archive