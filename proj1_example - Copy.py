import numpy as np
import string
import pandas as pd
import nltk
import keras
from sklearn import random_projection
from sklearn.metrics import accuracy_score
from nltk.corpus import stopwords
from keras.layers import merge
from keras.models import Sequential
from keras.layers import Embedding, Dense, Dropout, LSTM
from keras.models import Sequential, Model
from keras.layers import Concatenate, Dense, LSTM, Input, concatenate
from keras.optimizers import SGD
from keras import metrics
from tempfile import TemporaryFile


stop_words = set(stopwords.words('english') + list(string.punctuation))


# -------------- Helper Functions --------------
def tokenize(text):
    '''
    :param text: a doc with multiple sentences, type: str
    return a word list, type: list
    https://textminingonline.com/dive-into-nltk-part-ii-sentence-tokenize-and-word-tokenize
    e.g.
    Input: 'It is a nice day. I am happy.'
    Output: ['it', 'is', 'a', 'nice', 'day', 'i', 'am', 'happy']
    '''
    tokens = []
    for word in nltk.word_tokenize(text):
        word = word.lower()
        if word not in stop_words and not word.isnumeric():
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



def read_business_funny_and_useful(file_name):
    df = pd.read_csv(file_name)
    company = np.array([x for x in range(0,len(df['business_id']))])
    #count=0;
    #for i in range(len(df)):
    #    exist=False
    #    for j in range(df.index.get_loc(i)):
    #        if i==j:
    #            exist=True
    #            company[df.index.get_loc(i)]=company[df.index.get_loc(j)]
    #            exit
    #    if not exist:
    #        company[df.index.get_loc(i)]=int(count)
    #        count+=1

    numeric_vector = np.array([df['funny'].apply(int),df['funny'].apply(int)])
    numeric_vector = np.concatenate([[company],[numeric_vector[0]],[numeric_vector[1]]])
    numeric_vector= numeric_vector.T
    return numeric_vector


def load_data(input_length):
    # Load training data and vocab
    train_id_list, train_data_label, train_data_matrix, vocab = read_data("data/train.csv", input_length)
    K = max(train_data_label) + 1  # labels begin with 0

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
    input_length = 300
    embedding_size = 100
    hidden_size = 100
    batch_size = 128
    dropout_rate = 0.5
    learning_rate = 0.1
    total_epoch = 10
    npzfile=np.load('save_matrix.npz')
    
    #if npzfile is None:
    #    train_id_list, train_data_matrix, train_data_label, \
    #        valid_id_list, valid_data_matrix, valid_data_label, \
    #        test_id_list, test_data_matrix,_,vocab = load_data(input_length)
    #    np.savez('save_matrix', train_id_list, train_data_matrix, train_data_label, \
    #        valid_id_list, valid_data_matrix, valid_data_label, \
    #        test_id_list, test_data_matrix,  len(vocab))
    #else:
    train_id_list=npzfile['arr_0']
    train_data_matrix=npzfile['arr_1']
    train_data_label=npzfile['arr_2']
    valid_id_list=npzfile['arr_3']
    valid_data_matrix=npzfile['arr_4']
    valid_data_label=npzfile['arr_5']
    test_id_list=npzfile['arr_6']
    test_data_matrix=npzfile['arr_7']
    vocab_len=npzfile['arr_8'] 
    # Data shape
    N = train_data_matrix.shape[0]
    K = train_data_label.shape[1]
    if vocab_len is None:
        input_size = len(vocab)+2
    else:
        input_size=int(vocab_len)+2
    output_size = K

    # New model
    model = Sequential()

    # embedding layer and dropout
    # YOUR CODE HERE
    model.add(Embedding(input_dim=input_size, output_dim=embedding_size,
    input_length=input_length))
    model.add(Dropout(dropout_rate))

    # LSTM layer
    # YOUR CODE HERE
    model.add(LSTM(units=hidden_size))

    # output layer
    # YOUR CODE HERE
    model.add(Dense(K, activation='softmax'))


    #train_data = pd.read_csv("data/train.csv")
    #train_data_label = np.array(train_data['stars'].apply(int) - 1)
    #valid_data = pd.read_csv("data/valid.csv")
    #valid_data_label = np.array(valid_data['stars'].apply(int) - 1)
    #test_data = pd.read_csv("data/test.csv")
    #test_data_label = np.array(valid_data['stars'].apply(int) - 1)

    #merge numeric training model into the model
    train_data_numeric = read_business_funny_and_useful("data/train.csv")
    valid_data_numeric = read_business_funny_and_useful("data/valid.csv")
    test_data_numeric = read_business_funny_and_useful("data/test.csv")
    model_numeric = Sequential()
    model_numeric.add(Dense(64, activation='relu', input_dim=3))
    model_numeric.add(Dropout(0.5))
    model_numeric.add(Dense(64, activation='relu'))
    model_numeric.add(Dropout(0.5))
    model_numeric.add(Dense(5, activation='softmax'))
    optimizer = SGD(lr=learning_rate, decay=1e-6, momentum=0.9, nesterov=True)
    model_numeric.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    model_numeric.fit(train_data_numeric, train_data_label, epochs=total_epoch, batch_size=batch_size)
    print(model_numeric.summary())
    #train_data_label = keras.utils.to_categorical(train_data_label, num_classes=5)
    #valid_data_label = keras.utils.to_categorical(valid_data_label, num_classes=5)
    #test_data_label = keras.utils.to_categorical(test_data_label, num_classes=5)
    # SGD optimizer with momentum
    #compile model
    merged=Concatenate([model, model_numeric])
    final_model = Sequential()
    final_model.add(merged)
    final_model.add(Dense(1, activation='softmax',input_shape=(N+5,)))
    final_model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    # training
    final_model.fit([train_data_matrix,train_data_numeric], train_data_label, epochs=total_epoch, batch_size=batch_size)
    # testing
    train_score = final_model.evaluate([train_data_matrix,train_data_numeric], train_data_label, batch_size=batch_size)
    print('Training Loss: {}\n Training Accuracy: {}\n'.format(trsain_score[0], train_score[1]))
    valid_score = final_model.evaluate([valid_data_matrix,valid_data_numeric], valid_data_label, batch_size=batch_size)
    print('Validation Loss: {}\n Validation Accuracy: {}\n'.format(valid_score[0], valid_score[1]))

    # predicting
    test_pre = final_model.predict([test_data_matrix,test_data_numeric], batch_size=batch_size).argmax(axis=-1) + 1
    sub_df = pd.DataFrame()
    sub_df["review_id"] = test_id_list
    sub_df["pre"] = test_pre
    sub_df.to_csv("pre.csv", index=False)
