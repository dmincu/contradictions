from __future__ import print_function

import csv
import getopt
import numpy as np
import pandas as pd
import re
import sys
import time

from gensim.models import word2vec
from keras.regularizers import l1, activity_l1
from keras.layers import merge, Convolution1D, Convolution2D, MaxPooling1D, MaxPooling2D, Input, Dense, Flatten
from keras.models import Model


FULL_CSV_PATH_DEV = '../dataset/snli_1.0/snli_1.0_dev.txt'
FULL_CSV_PATH_TRAIN = '../dataset/snli_1.0_train.txt_Pieces/snli_1.0_train_'
FULL_CSV_PATH_TEST = '../dataset/snli_1.0/snli_1.0_test.txt'
FULL_MODEL_PATH = '../dataset/GoogleNews-vectors-negative300.bin'
FULL_MODEL_PATH_DEV = '../dataset/text8'

FILE = sys.stdout

MAX_SIZE_SENTENCE = 10
MAX_EMBEDDINGS_SIZE = 200
MODEL = None


def get_dataframe_from_csv(csv_path):
    infile = open(csv_path, 'rb')
    records = csv.DictReader(infile, delimiter='\t', skipinitialspace=True)

    result_list = []
    for result in records:
        result_list.append(result)

    df = pd.DataFrame.from_dict(result_list)

    return df


def add_padding_to_sentence_embeddings(sentence_embeddings):
    no_words = len(sentence_embeddings)
    if no_words > 0:
        embeddings_size = len(sentence_embeddings[0])
    else:
        return [[0] * MAX_EMBEDDINGS_SIZE] * MAX_SIZE_SENTENCE
    padding = [[0] * embeddings_size] * (MAX_SIZE_SENTENCE - no_words)

    return sentence_embeddings + padding


def get_embedding_for_entry(entry):
    words = re.findall(r"\w+", entry)
    embeddings = [
        MODEL[x] for x in words if x in MODEL.vocab
    ]

    embeddings = add_padding_to_sentence_embeddings(embeddings)
    return embeddings


def get_embeddings_lists(df, label):
    results_s1 = df[label].apply(get_embedding_for_entry)
    results_s1_array = np.array([x for x in results_s1[:]])

    return np.array([results_s1_array])


def get_target_values(df):
    labels = {
        'contradiction': 1,
        'neutral': 2,
        '-': 3,
        'entailment': 4
    }
    return df['gold_label'].apply(lambda x: labels[x]).values


def train_model(df):
    inputs_s1 = get_embeddings_lists(df, 'sentence1')
    inputs_s2 = get_embeddings_lists(df, 'sentence2')
    labels = get_target_values(df)

    (dim_1, channels, dim_2, dim_3) = inputs_s1.shape

    print(dim_1)
    print(dim_2)
    print(dim_3)

    # Establish the input
    net_input = Input(shape=(channels, dim_2, dim_3))
    x = Convolution2D(200, 5, 3, W_regularizer=l1(0.01))(net_input)
    x = Convolution2D(200, 5, 2, W_regularizer=l1(0.01))(x)
    x = MaxPooling2D(2, 2)(x)
    out = Flatten()(x)

    contradiction_model = Model(net_input, out)

    input_a = Input(shape=(channels, dim_2, dim_3))
    input_b = Input(shape=(channels, dim_2, dim_3))

    out_a = contradiction_model(input_a)
    out_b = contradiction_model(input_b)

    concatenated = merge([out_a, out_b], mode='concat')
    out = Dense(1, activation='softmax')(concatenated)

    classification_model = Model([input_a, input_b], out)

    classification_model.compile(
        optimizer='rmsprop',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    classification_model.fit(
        [inputs_s1, inputs_s2],
        labels,
        nb_epoch=10,
        batch_size=100
    )

    print('it works to train this thing!')


if __name__ == '__main__':
    # Parse command line arguments
    try:
        opts, args = getopt.getopt(
            sys.argv[1:],
            'm:',
            [
                'ni=',
                'no=',
                'batch_size=',
                'chunk_no=',
                'print_garbage',
                'use_file',
            ]
        )
    except getopt.GetoptError:
        print(
            'test.py -m <method> --ni <no_input> --no <no_output>' +
            ' [--print_garbage, --use_file --use_unigrams --use_all_lexical]' +
            ', where method [classic, svm, nb]',
            file=FILE
        )
        sys.exit(2)

    method = 'lstm'
    print_garbage = False
    use_file = False
    use_dev = False
    ni = 1
    no = 100
    batch_size = 10
    chunk_no = 27

    for opt, arg in opts:
        if opt == '-m':
            method = arg
        elif opt == '--ni':
            ni = int(arg)
        elif opt == '--no':
            no = int(arg)
        elif opt == '--chunk_no':
            chunk_no = int(arg)
        elif opt == '--print_garbage':
            print_garbage = True
        elif opt == '--use_file':
            use_file = True
        elif opt == '--use_dev':
            use_dev = True
        elif opt == '--batch_size':
            batch_size = int(arg)

    # Make input file name
    FULL_CSV_PATH_TRAIN += str(chunk_no) + '.txt'

    # Open output file
    if use_file:
        FILE = open(
            '../results/' + time.strftime("%d_%m_%Y_%H_%M_%S") +
            '_' + method + '_ni_' + str(ni) + '_no_' + str(no) + '_.txt',
            'w'
        )

    # Do the magic
    if use_dev:
        df_train = get_dataframe_from_csv(FULL_CSV_PATH_DEV)
    else:
        df_train = get_dataframe_from_csv(FULL_CSV_PATH_TRAIN)
    df_test = get_dataframe_from_csv(FULL_CSV_PATH_TEST)

    #sentences = word2vec.Text8Corpus(FULL_MODEL_PATH_DEV)
    #MODEL = word2vec.Word2Vec(sentences, size=200)
    sentences = [['first', 'sentence'], ['second', 'sentence']]
    MODEL = word2vec.Word2Vec()
    MODEL.build_vocab(sentences)
    # MODEL = word2vec.Word2Vec.load_word2vec_format(
    #    FULL_MODEL_PATH,
    #    binary=True
    # )
    # print([MODEL[x] for x in 'this model hus everything'.split() if x in MODEL.vocab])

    train_model(df_train)

    # Close output file
    FILE.close()
