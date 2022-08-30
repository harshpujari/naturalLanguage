#link for dataset 
#https://www.kaggle.com/competitions/sentiment-analysis-on-movie-reviews/rules

#installing missing library
!pip install transformers

#importing libraries 
import pandas as pd
import numpy as np
import tensorflow as tf
from transformers import BertTokenizer
from transformers import TFAutoModel

#reading dataset
df = pd.read_csv('train.tsv', sep='\t')

#removing unwanted columns
df = df[['Phrase', 'Sentiment']]

#if you have low processing capacity , frop few rows
#df.drop([5,6], axis=0, inplace=True)

df.shape

#here pretrained tokenizer was used
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

#tokenizing the sentence 
def tokenize(sentence):
    tokens = tokenizer.encode_plus(sentence, max_length=512,
                                   truncation=True, padding='max_length',
                                   add_special_tokens=True, return_token_type_ids=False,
                                   return_tensors='tf')
    return tokens['input_ids'], tokens['attention_mask']

#creaing np.arrays which can be used to add id and mask in future
Xids = np.zeros((len(df), 512))
Xmask = np.zeros((len(df), 512))

for i, sequence in enumerate(df['Phrase']):
    tokens = tokenize(sequence)
    Xids[i, :], Xmask[i, :] = tokens[0], tokens[1]

arr = df['Sentiment'].values

labels = np.zeros((arr.size, arr.max()+1))

labels[np.arange(arr.size), arr] = 1

with open('movie-xids.npy', 'wb') as f:
    np.save(f, Xids)
with open('movie-xmask.npy', 'wb') as f:
    np.save(f, Xmask)
with open('movie-labels.npy', 'wb') as f:
    np.save(f, labels)

del df, Xids, Xmask, labels

with open('movie-xids.npy', 'rb') as f:
    Xids = np.load(f, allow_pickle=True)
with open('movie-xmask.npy', 'rb') as f:
    Xmask = np.load(f, allow_pickle=True)
with open('movie-labels.npy', 'rb') as f:
    labels = np.load(f, allow_pickle=True)

tf.config.experimental.list_physical_devices('GPU')

data = tf.data.Dataset.from_tensor_slices((Xids, Xmask, labels))  # [750000:850000]

SHUFFLE = 100000
BATCH_SIZE = 16

def map_func(input_ids, masks, labels):
    return {'input_ids': input_ids, 'attention_mask': masks}, labels

data = data.map(map_func)

data = data.shuffle(SHUFFLE).batch(BATCH_SIZE)

SIZE = Xids.shape[0]/BATCH_SIZE

SPLIT = 0.9

train = data.take(int(SIZE*SPLIT))
val = data.skip(int(SIZE*SPLIT))

del data

bert = TFAutoModel.from_pretrained('bert-base-cased')  #, output_hidden_states=False

bert.summary()

input_ids = tf.keras.layers.Input(shape=(512,), name='input_ids', dtype='int32')
mask = tf.keras.layers.Input(shape=(512,), name='attention_mask', dtype='int32')

embeddings = bert.bert(input_ids, attention_mask=mask)[0]

x = tf.keras.layers.Dropout(0.1)(embeddings)
x = tf.keras.layers.GlobalMaxPool1D()(x)
y = tf.keras.layers.Dense(5, activation='softmax', name='outputs')(x)

model = tf.keras.Model(inputs=[input_ids, mask], outputs=y)

model.layers[2].trainable = False

model.summary()

optimizer = tf.keras.optimizers.Adam(lr=0.01, decay=1e-6)
loss = tf.keras.losses.CategoricalCrossentropy()
acc = tf.keras.metrics.CategoricalAccuracy('accuracy')

model.compile(optimizer=optimizer, loss=loss, metrics=[acc])

# 800K
history = model.fit(
    train,
    validation_data=val,
    epochs=30)

model.get_config()

model.save('sentiment_model')

del model

model = tf.keras.models.load_model('sentiment_model')
model.summary()

loss, acc = model.evaluate(val)

val.take(1)

pd.set_option('display.max_colwidth', None)
df = pd.read_csv('test.tsv', sep='\t')
df.head()

df = df.drop_duplicates(subset=['SentenceId'], keep='first')
df.head()

#df.drop(df.index[200:], inplace=True)

tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

def prep_data(text):
    tokens = tokenizer.encode_plus(text, max_length=512,
                                   truncation=True, padding='max_length',
                                   add_special_tokens=True, return_token_type_ids=False,
                                   return_tensors='tf')
    # tokenizer returns int32 tensors, we need to return float64, so we use tf.cast
    return {'input_ids': tf.cast(tokens['input_ids'], tf.float64),
            'attention_mask': tf.cast(tokens['attention_mask'], tf.float64)}

probs = model.predict(prep_data("bad, worst, ugly"))[0]
np.argmax(probs)

df['Sentiment'] = None

for i, row in df.iterrows():
    # get token tensors
    tokens = prep_data(row['Phrase'])
    # get probabilities
    probs = model.predict(tokens)
    # find argmax for winning class
    pred = np.argmax(probs)
    # add to dataframe
    df.at[i, 'Sentiment'] = pred

df.sample(20)
