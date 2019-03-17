# Databricks notebook source
import numpy as np
import string
import pandas as pd
import nltk
import keras

from sklearn import random_projection
from sklearn.metrics import accuracy_score
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize

from keras.models import Model, load_model
from keras.layers import Embedding, Dense, Dropout, LSTM, Input, Bidirectional, GRU, Activation, BatchNormalization
from keras.layers import Conv1D, Flatten, MaxPooling1D, AveragePooling1D, Lambda, GlobalMaxPooling1D, Conv2D, MaxPooling2D
from keras.layers.merge import Concatenate
from keras.optimizers import SGD, Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import metrics, regularizers
from keras.utils import plot_model

from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier

stop_words = set(stopwords.words('english') + list(string.punctuation))


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

def read_business_funny_and_useful(file_name):
    df = pd.read_csv(file_name)
    #company = np.array([x for x in range(0,len(df['business_id']))])
    #for i in company:
    #    if len(np.where(companyid==companyid[i])[0])>1:
    #        i=np.where(companyid==companyid[i])[0][0]
    companyid = np.array(df['business_id'])
    company_count=[len(np.where(companyid==companyid[i])[0]) for i in range(0,len(companyid))]
    numeric_vector = np.array([df['funny'].apply(int),df['funny'].apply(int),df['cool'].apply(int)])
    numeric_vector = np.concatenate([[company_count],[numeric_vector[0]],[numeric_vector[1]],[numeric_vector[2]]])
    numeric_vector= numeric_vector.T
    return numeric_vector




def create_model(dropout_rate = 0.2, filter_size = 100, embedding_size = 100, hidden_size = 100):

    #################### RCNN Model #########################
    # define two sets of inputs
    inputB = Input(shape=(4,))
    model_numeric_Dense=Dense(64, activation='relu', input_dim=3)(inputB)
    model_numeric_Drop=(Dropout(0.5))(model_numeric_Dense)
    model_numeric_Dense2=(Dense(64, activation='relu'))(model_numeric_Drop)
    model_numeric_Drop2=(Dropout(0.5))(model_numeric_Dense2)
    model_numeric_Dense3=(Dense(5, activation='softmax'))(model_numeric_Drop2)

    x = Input(shape=(input_length,))
    emb = Embedding(input_dim=input_size, output_dim=embedding_size, input_length=input_length)(x)
    dr_emb = Dropout(dropout_rate)(emb)

    conv_1 = Conv1D(filters=16, kernel_size=7, padding='same', kernel_regularizer=regularizers.l2(reg))(dr_emb)
    bn_conv_1 = BatchNormalization()(conv_1)
    act_conv_1 = Activation('tanh')(bn_conv_1)
    dr_conv_1 = Dropout(dropout_rate)(act_conv_1)
    gmp_conv_1 = MaxPooling1D()(dr_conv_1)

    conv_2 = Conv1D(filters=16, kernel_size=5, padding='same', kernel_regularizer=regularizers.l2(reg))(gmp_conv_1)
    bn_conv_2 = BatchNormalization()(conv_2)
    act_conv_2 = Activation('tanh')(bn_conv_2)
    dr_conv_2 = Dropout(dropout_rate)(act_conv_2)
    gmp_conv_2 = MaxPooling1D()(dr_conv_2)

    conv_3 = Conv1D(filters=32, kernel_size=3, padding='same', kernel_regularizer=regularizers.l2(reg))(gmp_conv_2)
    bn_conv_3 = BatchNormalization()(conv_3)
    act_conv_3 = Activation('tanh')(bn_conv_3)
    gmp_conv_3 = MaxPooling1D()(act_conv_3)

    conv_4 = Conv1D(filters=32, kernel_size=3, padding='same', kernel_regularizer=regularizers.l2(reg))(gmp_conv_3)
    bn_conv_4 = BatchNormalization()(conv_4)
    act_conv_4 = Activation('tanh')(bn_conv_4)
    dr_conv_4 = Dropout(dropout_rate)(act_conv_4)
    gmp_conv_4 = MaxPooling1D()(dr_conv_4)

    lstm = Bidirectional(LSTM(units=hidden_size_2, recurrent_dropout=0.2))(gmp_conv_4)

    gru = Bidirectional(GRU(units=hidden_size_2, recurrent_dropout=0.2))(dr_emb)
    merge = Concatenate(axis=-1)([lstm, gru])
    dr = Dropout(dropout_rate)(merge)
    d = Dense(125, activation='relu', kernel_regularizer=regularizers.l2(reg))(dr)
    d = Dense(25, activation='relu')(d)
    merge2 = Concatenate(axis=-1)([d, model_numeric_Dense3])
    y = Dense(output_size, activation='softmax')(merge2)
    model = Model([x,inputB], y)
    #################### RCNN Model #########################

    #print(model.summary())
    #plot_model(model, to_file='model_v6.png')
    optimizer=keras.optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, decay=0.0, amsgrad=False)

    # compile model
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model


if __name__ == '__main__':
    # Hyperparameters
    # Hyperparameters
    input_length = 320
    hidden_size = 100
    hidden_size_2 = np.int(hidden_size/2)
    batch_size = 100
    learning_rate = 0.001
    total_epoch = 10

    pool_size=2
    reg = 0.01
    #train_id_list, train_data_matrix, train_data_label, \
    #  valid_id_list, valid_data_matrix, valid_data_label, \
    #  test_id_list, test_data_matrix, _, vocab = load_data(input_length)
    #np.savez('save_matrix', train_id_list, train_data_matrix, train_data_label, \
    #    valid_id_list, valid_data_matrix, valid_data_label, \
    #    test_id_list, test_data_matrix,  len(vocab))
    npzfile=np.load('save_matrix.npz')

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
 
    #train_data_numeric = read_business_funny_and_useful("data/train.csv")
    #valid_data_numeric = read_business_funny_and_useful("data/valid.csv")
    #test_data_numeric = read_business_funny_and_useful("data/test.csv")
    #np.savez('save_numeric_matrix',train_data_numeric,valid_data_numeric,test_data_numeric)
    npzfile2=np.load('save_numeric_matrix.npz')

    train_data_numeric=npzfile2['arr_0']
    valid_data_numeric=npzfile2['arr_1']
    test_data_numeric=npzfile2['arr_2']
    model = create_model()

    es = EarlyStopping(monitor='val_loss', mode='min', patience=3)
    mc = ModelCheckpoint('best_model.h5', monitor='val_acc', mode='max', verbose=1,save_best_only=True)

    validation = ([valid_data_matrix,valid_data_numeric], valid_data_label)
    # training
    history = model.fit([train_data_matrix,train_data_numeric], train_data_label, validation_data=validation, epochs=total_epoch, batch_size=batch_size, callbacks=[es, mc])

    '''
    model = KerasClassifier(build_fn=create_model, epochs=2,  verbose=0)
    dropout_rate = [0.3, 0.5, 0.7]
    filter_size = [10, 50, 100, 200]
    embedding_size = [50, 100, 200]
    hidden_size = [50, 100, 200]

    param_grid = dict(dropout_rate=dropout_rate, filter_size=filter_size, embedding_size=embedding_size, hidden_size=hidden_size)
    grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1)
    grid_result = grid.fit(train_data_matrix, train_data_label)

    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))

    '''
    #plot_history(history)
    saved_model = load_model('best_model.h5')

    # testing
    train_score = saved_model.evaluate([train_data_matrix,train_data_numeric], train_data_label, batch_size=batch_size)
    print('Training Loss: {}\n Training Accuracy: {}\n'.format(train_score[0], train_score[1]))
    valid_score = saved_model.evaluate([valid_data_matrix,valid_data_numeric], valid_data_label, batch_size=batch_size)
    print('Validation Loss: {}\n Validation Accuracy: {}\n'.format(valid_score[0], valid_score[1]))

    # predicting
    #test_pre = model.predict(test_data_matrix, batch_size=batch_size).argmax(axis=-1) + 1
    #sub_df = pd.DataFrame()
    #sub_df["review_id"] = test_id_list
    #sub_df["pre"] = test_pre
    #sub_df.to_csv("pre_5.csv", index=False)
