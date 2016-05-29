from __future__ import print_function

import csv
import getopt
import pandas as pd
import re
import sys
import time

from gensim.models import word2vec



FULL_CSV_PATH_DEV = '../dataset/snli_1.0/snli_1.0_dev.txt'
FULL_CSV_PATH_TRAIN = '../dataset/snli_1.0_train.txt_Pieces/snli_1.0_train_'
FULL_CSV_PATH_TEST = '../dataset/snli_1.0/snli_1.0_test.txt'
FULL_MODEL_PATH = '../dataset/GoogleNews-vectors-negative300.bin'

FILE = sys.stdout


def get_dataframe_from_csv(csv_path):
    infile = open(csv_path, 'rb')
    records = csv.DictReader(infile, delimiter='\t', skipinitialspace=True)

    result_list = []
    for result in records:
        result_list.append(result)

    df = pd.DataFrame.from_dict(result_list)

    return df


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

    w2vmodel = word2vec.Word2Vec.load_word2vec_format(
        FULL_MODEL_PATH,
        binary=True
    )
    print([x for x in 'this model hus everything'.split() if x in w2vmodel.vocab])

    # Close output file
    FILE.close()
