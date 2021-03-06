import numpy as np
import string
import pandas as pd
import nltk
import keras
from keras import regularizers
from sklearn import random_projection
from sklearn.metrics import accuracy_score
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from keras.models import Sequential
from keras.layers import Embedding, Dense, Dropout, LSTM, GlobalMaxPooling1D
from keras.layers import Conv1D, Flatten, Activation, MaxPooling1D, Lambda, Flatten, Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras import metrics
from keras.layers.normalization import BatchNormalization
stop_words = set(stopwords.words('english') + list(string.punctuation))
from keras import initializers
from keras.layers import LeakyReLU
import random

#--------------- New Functions------------------
def is_negation(word):
    #input: string, output:boolean
    negation=['no','never','not','n\'t','cannot','aint']
    for neg in negation:
        if (word==neg):
            return True;
    return False

def replace_apostrophe(words):
    words=words.replace(',', '.')
    words=words.replace(';', '.')
    return words


def split(arr, size):
     arrs = []
     while len(arr) > size:
         pice = arr[:size]
         arrs.append(pice)
         arr   = arr[size:]
     arrs.append(arr)
     return arrs

# -------------- Helper Functions --------------

def tokenize(text):
    '''
    :param text: a doc with multiple sentences, type: str
    return a word list, type: list
    https://textminingonline.com/dive-into-nltk-part-ii-sentence-tokenize-and-word-tokenize
    e.g.
    Input: 'It is a nice day. I am happy.'
    Output: ['it', 'is', 'a', 'nice', 'day', 'i', 'am', 'happy']
    
    tokens = []
    for word in nltk.word_tokenize(text):
        word = word.lower()
        if word not in stop_words and not word.isnumeric():
            tokens.append(word)
    return tokens
    '''
    tokens = []
    new_text=replace_apostrophe(text)
    sent_tokenize_list = sent_tokenize(new_text)
    for sentence in sent_tokenize_list:
        negative=False
        for word in nltk.word_tokenize(sentence):
            word = word.lower()
            if is_negation(word):
                negative=not negative
                tokens.append(word)
            elif word not in stop_words and not word.isnumeric():
                if (negative):
                    word='not_'+word
                tokens.append(word)
    return tokens

def get_sequence(data, seq_length, vocab_dict):
    '''
    :param data: a list of words, type: list
    :param seq_length: the length of sequences,, type: int
    :param vocab_dict: a dict from words to indices, type: dict
    return a dense sequence matrix whose elements are indices of words,
    '''
    data_matrix = np.zeros((len(data), seq_length), dtype=int)
    for i, doc in enumerate(data):
        for j, word in enumerate(doc):
            # YOUR CODE HERE
            if j == seq_length:
                break
            word_idx = vocab_dict.get(word, 1) # 1 means the unknown word
            data_matrix[i, j] = word_idx
    return data_matrix




def read_data(file_name, input_length, vocab=None):
    """
    https://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_csv.html
    """
    df = pd.read_csv(file_name)
    df['words'] = df['text'].apply(tokenize)
    cool = df['cool'].apply(int)
    funny = df['funny'].apply(int)
    useful = df['useful'].apply(int)


    for i in range(df['words'].shape[0]):
        if (df['cool'][i]>=2):
            df['words'][i].append('cool:2');
        else:
            df['words'][i].append('cool:'+str(df['cool'][i]))
        df['words'][i].append('null')
        if (df['funny'][i]>=2):
            df['words'][i].append('funny:2');
        else:
            df['words'][i].append('funny:'+str(df['funny'][i]))
        df['words'][i].append('null')
        if (df['useful'][i]>=2):
            df['words'][i].append('useful:2');
        else:
            df['words'][i].append('useful:'+str(df['useful'][i]))
        df['words'][i].append('null')

    if vocab is None:
        vocab = set()
        for i in range(len(df)):
            for word in df.iloc[i]['words']:
                vocab.add(word)
    vocab_dict = dict()
    vocab_dict['<pad>'] = 0 # 0 means the padding signal
    vocab_dict['<unk>'] = 1 # 1 means the unknown word
    vocab_size = 2
    for v in vocab:
        vocab_dict[v] = vocab_size
        vocab_size += 1

    data_matrix = get_sequence(df['words'], input_length, vocab_dict)
    stars = df['stars'].apply(int) - 1    
    return df['review_id'], stars, data_matrix, vocab
# ----------------- End of Helper Functions-----------------


def load_data(input_length):
    # Load training data and vocab
    train_id_list, train_data_label, train_data_matrix, vocab = read_data("data/train.csv", input_length)
    K = max(train_data_label)+1  # labels begin with 0

    # Load valid data
    valid_id_list, valid_data_label, valid_data_matrix, vocab = read_data("data/valid.csv", input_length, vocab=vocab)

    # Load testing data
    test_id_list, _, test_data_matrix, _ = read_data("data/test.csv", input_length, vocab=vocab)
    
    print("Vocabulary Size:", len(vocab))
    print("Training Set Size:", len(train_id_list))
    print("Validation Set Size:", len(valid_id_list))
    print("Test Set Size:", len(test_id_list))
    print("Training Set Shape:", train_data_matrix.shape)
    print("Validation Set Shape:", valid_data_matrix.shape)
    print("Testing Set Shape:", test_data_matrix.shape)

    # Converts a class vector to binary class matrix.
    # https://keras.io/utils/#to_categorical
    train_data_label = keras.utils.to_categorical(train_data_label, num_classes=K)
    valid_data_label = keras.utils.to_categorical(valid_data_label, num_classes=K)
    return train_id_list, train_data_matrix, train_data_label, \
        valid_id_list, valid_data_matrix, valid_data_label, \
        test_id_list, test_data_matrix, None, vocab


if __name__ == '__main__':
    # Hyperparameters



    batch_size = 128
    input_length = 256
    embedding_size = 128
    total_epoch=4
    learning_rate=0.001
    
    train_id_list, train_data_matrix, train_data_label, \
        valid_id_list, valid_data_matrix, valid_data_label, \
        test_id_list, test_data_matrix, _, vocab = load_data(input_length)

    # Data shape
    N = train_data_matrix.shape[0]
    K = train_data_label.shape[1]

    input_size = len(vocab) + 2
    output_size = K
    print("output size",K)
    # New model

    
    for gs in range(20):
        learning_rate=random.uniform(1e-4,1e-3)
        batch_size=random.randint(32,256)

    ####################CNN-LTSM Model#########################
        model = Sequential()    
        model.add(Embedding(input_dim=input_size, output_dim=embedding_size, input_length=input_length))
    #    model.add(Dropout(0.2))
        model.add(LSTM(embedding_size, return_sequences=True, input_shape=(input_length, embedding_size), dropout=0.5, recurrent_dropout=0.5)) 
        model.add(Conv1D(filters=64, kernel_size=3, padding='same', activation='relu'))
        model.add(BatchNormalization())
        
        model.add(Conv1D(filters=64, kernel_size=3, padding='same', activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling1D(pool_size=2))    
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))  
        model.add(Dense(K, activation='softmax'))    
        model.summary()
        for layer in model.layers:
            print(layer.input_shape)
        optimizer=keras.optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        
    ####################CNN-LTSM Model#########################





        # Adam optimizer

        # compile model
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        train_data_matrix_split=split(train_data_matrix,20000)
        train_data_label_split=split(train_data_label,20000)

        print('learning rate: ',learning_rate)
        print('batch size: ',batch_size)
        # training
        best_acc=0
        for i in range(total_epoch):
            print("Epoch: ",i+1)
            print("current best validation accuracy",best_acc)
            for j in range(5):
                model.fit(train_data_matrix_split[j], train_data_label_split[j], epochs=1, batch_size=batch_size)
                valid_score = model.evaluate(valid_data_matrix, valid_data_label, batch_size=batch_size)
                print('Validation Loss: {}\n Validation Accuracy: {}\n'.format(valid_score[0], valid_score[1]))  
                if (valid_score[1]>best_acc):
                    best_acc=valid_score[1]
                    if (best_acc>0.6):
                        print('new validation accuracy record! Update prediction data...')
                        test_pre = model.predict(test_data_matrix, batch_size=batch_size).argmax(axis=-1) + 1
                        sub_df = pd.DataFrame()
                        sub_df["review_id"] = test_id_list
                        sub_df["pre"] = test_pre
                        sub_df.to_csv("pre.csv", index=False)
        

    # testing
    train_score = model.evaluate(train_data_matrix, train_data_label, batch_size=batch_size)
    print('Training Loss: {}\n Training Accuracy: {}\n'.format(train_score[0], train_score[1]))
    valid_score = model.evaluate(valid_data_matrix, valid_data_label, batch_size=batch_size)
    print('Validation Loss: {}\n Validation Accuracy: {}\n'.format(valid_score[0], valid_score[1]))

    # predicting
    test_pre = model.predict(test_data_matrix, batch_size=batch_size).argmax(axis=-1) + 1
    sub_df = pd.DataFrame()
    sub_df["review_id"] = test_id_list
    sub_df["pre"] = test_pre
    sub_df.to_csv("pre.csv", index=False)
