Exp.No : 06 
&emsp;
&emsp;
&emsp;
&emsp;
&emsp;
&emsp;
&emsp;
&emsp;
&emsp;
&emsp;
&emsp;
&emsp;
&emsp;
&emsp;
&emsp;
&emsp;
&emsp;
&emsp;
&emsp;
&emsp;
&emsp;
&emsp;
&emsp;
&emsp;
Date : 16.10.2023
<br>
# Named Entity Recognition using LSTM

## AIM

To develop an LSTM-based model for recognizing the named entities in the text.

## Problem Statement and Dataset
- We aim to develop an LSTM-based neural network model using Bidirectional Recurrent Neural Networks for recognizing the named entities in the text.
- The dataset used has a number of sentences, and each words have their tags.
- We have to vectorize these words using Embedding techniques to train our model.
- Bidirectional Recurrent Neural Networks connect two hidden layers of opposite directions to the same output.

<p align="left">
<img src="https://github.com/Kaushika-Anandh/named-entity-recognition/blob/main/1.PNG" width="150" height="300">
</p>

## Neural Network Model

<p align="center">
<img src="https://github.com/Kaushika-Anandh/named-entity-recognition/blob/main/2.jpg" width="650" height="450">
</p>


## DESIGN STEPS

-  **Step 1:** Import the necessary packages.
-  **Step 2:** Read the dataset, and fill the null values using forward fill.
-  **Step 3:** Create a list of words, and tags. Also find the number of unique words and tags in the dataset.
-  **Step 4:** Create a dictionary for the words and their Index values. Do the same for the tags as well,Now we move to moulding the data for training and testing.
-  **Step 5:** We do this by padding the sequences,This is done to acheive the same length of input data.
-  **Step 6:** We build a build a model using Input, Embedding, Bidirectional LSTM, Spatial Dropout, Time Distributed Dense Layers.
-  **Step 7:** We compile the model and fit the train sets and validation sets,We plot the necessary graphs for analysis, A custom prediction is done to test the model manually.


## PROGRAM
> Developed by: Kaushika A <br>
> Register no: 212221230048

**importing libraries**
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing import sequence

from keras import layers
from keras.models import Model
```

**loading dataset**
```python
data = pd.read_csv("ner_dataset.csv", encoding="latin1")
data.head(50)

data = data.fillna(method="ffill")
data.head(50)
```

**list of unique word and tag**
```python
print("Unique words in corpus:", data['Word'].nunique())
print("Unique tags in corpus:", data['Tag'].nunique())

words=list(data['Word'].unique())
words.append("ENDPAD")
num_words = len(words)
tags=list(data['Tag'].unique())
num_tags = len(tags)

print("Unique tags are:", tags)
```

**grouping by sentence class**
```python
class SentenceGetter(object):
    def __init__(self, data):
        self.n_sent = 1
        self.data = data
        self.empty = False
        agg_func = lambda s: [(w, p, t) for w, p, t in zip(s["Word"].values.tolist(),
                                                           s["POS"].values.tolist(),
                                                           s["Tag"].values.tolist())]
        self.grouped = self.data.groupby("Sentence #").apply(agg_func)
        self.sentences = [s for s in self.grouped]
    
    def get_next(self):
        try:
            s = self.grouped["Sentence: {}".format(self.n_sent)]
            self.n_sent += 1
            return s
        except:
            return None
```
```python
getter = SentenceGetter(data)
sentences = getter.sentences
len(sentences)

word2idx = {w: i + 1 for i, w in enumerate(words)}
tag2idx = {t: i for i, t in enumerate(tags)}
X1 = [[word2idx[w[0]] for w in s] for s in sentences]
```

**x and y data**
```python
max_len = 100
X = sequence.pad_sequences(maxlen=max_len,
                  sequences=X1, padding="post",
                  value=num_words-1)

y1 = [[tag2idx[w[2]] for w in s] for s in sentences]

y = sequence.pad_sequences(maxlen=max_len,
                  sequences=y1,
                  padding="post",
                  value=tag2idx["O"])

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2, random_state=1)
```

**network model**
```python
input_word = layers.Input(shape=(max_len,))

embedding_layer = layers.Embedding(input_dim=num_words,output_dim=50,
                                   input_length=max_len)(input_word)

dropout = layers.SpatialDropout1D(0.1)(embedding_layer)

bid_lstm = layers.Bidirectional(
    layers.LSTM(units=100,return_sequences=True,
                recurrent_dropout=0.1))(dropout)

output = layers.TimeDistributed(
    layers.Dense(num_tags,activation="softmax"))(bid_lstm)

model = Model(input_word, output)  

model.summary()

model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

model.fit(x=X_train, y=y_train, validation_data=(X_test,y_test),
          batch_size=50, epochs=3,)
```

**metrics**
```python
metrics = pd.DataFrame(history.history)
metrics.head()

metrics[['accuracy','val_accuracy']].plot()
metrics[['loss','val_loss']].plot()
```

**prediction for single input**
```python
i = 20
p = model.predict(np.array([X_test[i]]))
p = np.argmax(p, axis=-1)
y_true = y_test[i]
print("{:15}{:5}\t {}\n".format("Word", "True", "Pred"))
print("-" *30)
for w, true, pred in zip(X_test[i], y_true, p[0]):
    print("{:15}{}\t{}".format(words[w-1], tags[true], tags[pred]))
```

## OUTPUT

### Training Loss, Validation Loss Vs Iteration Plot
<p float="left">
  <img  src="https://github.com/Kaushika-Anandh/named-entity-recognition/blob/main/3.png" width="300" height="200">
  <img  src="https://github.com/Kaushika-Anandh/named-entity-recognition/blob/main/4.png" width="300" height="200">
</p>


### Sample Text Prediction
<img  src="https://github.com/Kaushika-Anandh/named-entity-recognition/blob/main/5.png" width="150" height="250">

## RESULT
Thus, an LSTM-based model for recognizing the named entities in the text is successfully developed.

