# Summary 

- What is a major drawback of word-based training for text generation instead of character-based generation?
    + Because there are far more words in a typical corpus than characters, so it is much more memory intensive. 



## Week-4 Exercise Shakespeare


```python
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
import tensorflow.keras.utils as ku 
import numpy as np 
```

```python
tokenizer = Tokenizer()
!wget --no-check-certificate \
    https://storage.googleapis.com/laurencemoroney-blog.appspot.com/sonnets.txt \
    -O /tmp/sonnets.txt
data = open('/tmp/sonnets.txt').read()

corpus = data.lower().split("\n")

# create the dictionary of words in the overall corpus
# with key being word, and value being token 
tokenizer.fit_on_texts(corpus)
# add 1 to total to consider OOV
total_words = len(tokenizer.word_index) + 1

# create input sequences using list of tokens
# something like that
# Line                          Input Sequences:
# [4 2 66 8 67 68 69 70]        [4 2]
#                               [4 2 66]
#                               [4 2 66 8]
#                               [4 2 66 8 67]
#                               [4 2 66 8 67 68]
#                               [4 2 66 8 67 68 69]
#                               [4 2 66 8 67 68 70]
input_sequences = []
for line in corpus:
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)


# find the max length of sequences
# pad sequences 
# Line                          Input Sequences[X+Y]:
# [4 2 66 8 67 68 69 70]        [0 0 0 0 0 0 0 0 0 0 4 2]
# [4 2 66 8 67 68 69 70]        [0 0 0 0 0 0 0 0 0 4 2 66]
# [4 2 66 8 67 68 69 70]        [0 0 0 0 0 0 0 0 4 2 66 67]
# [4 2 66 8 67 68 69 70]        [0 0 0 0 0 0 0 4 2 66 67 68]
# [4 2 66 8 67 68 69 70]        [0 0 0 0 0 0 4 2 66 67 68 69]
# [4 2 66 8 67 68 69 70]        [0 0 0 0 0 4 2 66 67 68 69 70]
max_sequence_len = max([len(x) for x in input_sequences])
input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))

# Now, that we have our sequences, the next thing we need to do is turn them into x's and y's, our input values and their labels. When you think about it, now that the sentences are represented in this way, all we have to do is take all but the last character as the x and then use the last character as the y on our label. We do that like this, where for the first sequence, everything up to the four is our input and the two is our label. 
# create predictors and label
predictors, label = input_sequences[:,:-1],input_sequences[:,-1]

# convert the Ys to one-hot encoding
label = ku.to_categorical(label, num_classes=total_words)
```

```python
model = Sequential()

# you can update the model hyper-parameters to make it a little bit better 
# with a larger corpous of work
# like embedding dim (100), 
model.add(Embedding(total_words, 100, input_length=max_sequence_len-1))
# create a bidiretional LSTM with 150 units in which big dog and dog big both makes sense.
model.add(Bidirectional(LSTM(150, return_sequences = True)))
model.add(Dropout(0.2))
model.add(LSTM(100))
# This layer will have one neuron, per word and that neuron should light up when we predict a given word.
model.add(Dense(total_words/2, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
model.add(Dense(total_words, activation='softmax'))
# set the learning rate to Adam optimizer
adam = Adam(lr, 0.01)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
print(model.summary())
```

` history = model.fit(predictors, label, epochs=100, verbose=1)`

```python
import matplotlib.pyplot as plt
acc = history.history['acc']
loss = history.history['loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'b', label='Training accuracy')
plt.title('Training accuracy')

plt.figure()

plt.plot(epochs, loss, 'b', label='Training Loss')
plt.title('Training loss')
plt.legend()

plt.show()
```

```python
seed_text = "Help me Obi Wan Kenobi, you're my only hope"
next_words = 100
  
for _ in range(next_words):
    # [x x x x x xx x x x  x x]
    token_list = tokenizer.texts_to_sequences([seed_text])[0]
    # crop off the last word of each sentences to get the label
    token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
    # give the token of the word most likely to be the next one in the sequence
    predicted = model.predict_classes(token_list, verbose=0)
    output_word = ""
    # iterate to find the word of that token & concatenate the sentence
    for word, index in tokenizer.word_index.items():
        if index == predicted:
            output_word = word
            break
    seed_text += " " + output_word
print(seed_text)
```

## Limitations to word-based RNN

Now, this approach works very well until you have very large bodies of text with many many words. So for example, you could try the complete works of Shakespeare and you'll likely hit memory errors, as assigning the one-hot encodings of the labels to matrices that have over `31,477` elements, which is the number of unique words in the collection, and there are over 15 million sequences generated using the algorithm that we showed here. So the labels alone would require the storage of many terabytes of RAM. So for your next task, you'll go through a workbook by yourself that uses character-based prediction. The full number of unique characters in a corpus is far less than the full number of unique words, at least in English. So the same principles that you use to predict words can be used to apply here. The workbook is at this URL, so try it out, and once you've done, that you'll be ready for this week's final exercise.


## Character-based RNN
[link](https://github.com/tensorflow/docs/blob/master/site/en/tutorials/text/text_generation.ipynb)

### The prediction task
Given a character, or a sequence of characters, what is the most probable next character? This is the task we're training the model to perform. The input to the model will be a sequence of characters, and we train the model to predict the output—the following character at each time step.
