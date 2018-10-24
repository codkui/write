'''Example script to generate text from Nietzsche's writings.
At least 20 epochs are required before the generated text
starts sounding coherent.
It is recommended to run this script on GPU, as recurrent
networks are quite computationally intensive.
If you try this script on new data, make sure your corpus
has at least ~100k characters. ~1M is better.
'''

from __future__ import print_function
from keras.callbacks import LambdaCallback
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
from keras.models import load_model
import numpy as np
import random
import sys
import io
import os

# path = get_file('data/tianlong.txt', origin='https://s3.amazonaws.com/text-datasets/nietzsche.txt')
path = 'data/lucongjinye.txt'
with io.open(path, encoding='utf-8') as f:
    text = f.read().lower()
print('corpus length:', len(text))

chars = sorted(list(set(text)))
print(chars[:240])
print('total chars:', len(chars))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

# cut the text in semi-redundant sequences of maxlen characters
oldLoss=0
lossAddNum=0
maxlen = 40
step = 3
sentences = []
next_chars = []
for i in range(0, len(text) - maxlen, step):
    sentences.append(text[i: i + maxlen])
    next_chars.append(text[i + maxlen])
print('nb sequences:', len(sentences))

print('Vectorization...')
x = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        x[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1


# build the model: a single LSTM
print('Build model...')
if os.path.exists("my_model.h5") and os.path.exists("my_model_weights.h5"):
    model = load_model('my_model.h5')
    model.load_weights('my_model_weights.h5')
else:
    model = Sequential()
    model.add(LSTM(128, input_shape=(maxlen, len(chars))))
    model.add(Dense(300))
    # model.add(Dense(300))
    #model.add(Dense(1300))
    model.add(Dense(len(chars)))
    
    model.add(Activation('softmax'))
    
optimizer = RMSprop(lr=0.00001)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)

def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


def on_epoch_end(epoch, logs):
    # Function invoked at end of each epoch. Prints generated text.
    
    global oldLoss
    global lossAddNum
    f= open("lstm.log", 'a+')
    print()
    print('----- Generating text after Epoch: %d' % epoch)
    # print(logs)
    if logs["loss"]>oldLoss:
        lossAddNum+=1
    else:
        lossAddNum=0
    oldLoss=logs["loss"]
    if lossAddNum>=4:
        exit()
    model.save('my_model.h5')
    model.save_weights('my_model_weights.h5')
    f.write('----- Generating loss: '+str(logs["loss"])+"\n")
    f.write('----- Generating text after Epoch: '+str(epoch)+"\n")
    
    start_index = random.randint(0, len(text) - maxlen - 1)
    for diversity in [0.2, 0.5, 1.0, 1.2]:
        
        print('----- diversity:', diversity)
        f.write('----- diversity:'+str(diversity)+"\n")
        generated = ''
        sentence = text[start_index: start_index + maxlen]
        generated += sentence
        print('----- Generating with seed: "' + sentence + '"')
        f.write('----- Generating with seed: "'+str(sentence)+"\n")
        sys.stdout.write(generated)
        f.write(str(generated))
        for i in range(400):
            x_pred = np.zeros((1, maxlen, len(chars)))
            for t, char in enumerate(sentence):
                x_pred[0, t, char_indices[char]] = 1.

            preds = model.predict(x_pred, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_char = indices_char[next_index]

            generated += next_char
            sentence = sentence[1:] + next_char

            sys.stdout.write(next_char)
            f.write(str(next_char))
            sys.stdout.flush()
        f.write("\n")
        print()
    f.close()
print_callback = LambdaCallback(on_epoch_end=on_epoch_end)

model.fit(x, y,
          batch_size=128,
          epochs=600,
callbacks=[print_callback])