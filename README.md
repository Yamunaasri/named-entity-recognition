# Named Entity Recognition

## AIM

To develop an LSTM-based model for recognizing the named entities in the text.

## Problem Statement and Dataset
we propose a deep neural network (NN) architecture, namely the bidirectional Long-Short Term Memory (Bi-LSTM) based model for NER. Based on existing gold standard datasets, we evaluated and compared several models for identifying biomedical named entities. Our deep NN based Bi-LSTM model using word and character level embeddings outperforms CRF and Bi-LSTM using only word level embeddings significantly.

![image](https://user-images.githubusercontent.com/75235813/197328754-b8dde19c-ac9e-4037-b154-907074372a7c.png)


## Neural Network Model

Include the neural network model diagram.

## DESIGN STEPS

-  **Step 1:** Import the necessary packages.
-  **Step 2:** Read the dataset, and fill the null values using forward fill.
-  **Step 3:** Create a list of words, and tags. Also find the number of unique words and tags in the dataset.
-  **Step 4:** Create a dictionary for the words and their Index values. Do the same for the tags as well,Now we move to moulding the data for training and testing.
-  **Step 5:** We do this by padding the sequences,This is done to acheive the same length of input data.
-  **Step 6:** We build a build a model using Input, Embedding, Bidirectional LSTM, Spatial Dropout, Time Distributed Dense Layers.
-  **Step 7:** We compile the model and fit the train sets and validation sets,We plot the necessary graphs for analysis, A custom prediction is done to test the model manually.

## PROGRAM
```Name : Yamunaasri T S
Reg no: 212222240117
```
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
![Screenshot 2024-04-22 151929](https://github.com/Yamunaasri/named-entity-recognition/assets/115707860/699279d0-b252-4d5f-b88c-bd434f07d1ba)

## RESULT
Thus, an LSTM-based model for recognizing the named entities in the text is successfully developed.
