from __future__ import print_function

import csv
import getopt
import itertools
import numpy as np
import pandas as pd
import re
import simplejson
import sys
import time

from collections import defaultdict
from itertools import islice, izip
from nltk.translate import bleu_score
from sklearn import svm

FULL_CSV_PATH_DEV = '../dataset/snli_1.0/snli_1.0_dev.txt'
FULL_CSV_PATH_TRAIN = '../dataset/snli_1.0/snli_1.0_train.txt'
FULL_CSV_PATH_TEST = '../dataset/snli_1.0/snli_1.0_test.txt'

ADJECTIVE = ['JJ', 'JJR', 'JJS']
NOUN = ['NN', 'NNS', 'NNP', 'NNPS']
VERB = ['VB', 'VBD', 'VBP', 'VBG', 'VBN', 'VBZ']
ADVERB = ['RB', 'RBR', 'RBS']

FILE = sys.stdout

FIELDS_TO_USE = []

#
# Helper functions
#


def get_dataframe_from_csv(csv_path):
    infile = open(csv_path, 'rb')
    records = csv.DictReader(infile, delimiter='\t', skipinitialspace=True)

    result_list = []
    for result in records:
        result_list.append(result)

    df = pd.DataFrame.from_dict(result_list)

    return df


def get_dataframe_subset_for_label(df, label):
    return df[df['gold_label'] == label]


def get_POStags_for_sentence(parse):
    tagged_words = {}

    for element in re.findall(r"([\w']+ [\w']+)", parse):
        split_element = element.split(' ')
        if (
            split_element[1].lower() in tagged_words and
            split_element[0] not in tagged_words[split_element[1].lower()]
        ):
            tagged_words[split_element[1].lower()].append(split_element[0])
        else:
            tagged_words[split_element[1].lower()] = [split_element[0]]

    return tagged_words


def count_overlap_in_words_list(premise_list, hypothesis_list):
    count = 0

    premise_list = map(lambda x: x.lower(), premise_list)
    hypothesis_list = map(lambda x: x.lower(), hypothesis_list)

    for word in premise_list:
        if word in hypothesis_list:
            count += 1

    return count


def filter_by_labels(tags, labels):
    return list(
        key for key, value in tags.iteritems()
        if len([x for x in value if x in labels]) > 0
    )


def sum_dicts(x, y):
    return {k: x.get(k, 0) + y.get(k, 0) for k in set(x) | set(y)}


def total_dif_first_dict_then_second(x, y):
    aux_dict = {k: abs(x.get(k, 0) - y.get(k, 0)) for k in set(x)}
    return sum(aux_dict.itervalues())


def sum_dataframe_dict(df, label):
    sum_dict = defaultdict(int)
    for x in df[label]:
        sum_dict = sum_dicts(x, sum_dict)
    sum_dict['total_count'] = sum(sum_dict.itervalues())
    return sum_dict


def sum_dataframe_dict_no_total(df, label):
    sum_dict = defaultdict(int)
    for x in df[label]:
        sum_dict = sum_dicts(x, sum_dict)
    return sum_dict


#
# Unlexicalized features
#


def get_bleu_score(premise, hypothesis, n):
    try:
        score = float(bleu_score.modified_precision(
            [re.findall(r"\w+", hypothesis)],
            re.findall(r"\w+", premise),
            n
        ))
        return score
    except ZeroDivisionError:
        return 0.0


def get_length_difference(premise, hypothesis):
    return abs(len(premise) - len(hypothesis))


def get_absolute_count_all(premise, hypothesis):
    return count_overlap_in_words_list(premise, hypothesis)


def get_absolute_count_label(premise_parse, hypothesis_parse, label):
    premise = filter_by_labels(premise_parse, label)
    hypothesis = filter_by_labels(hypothesis_parse, label)

    return count_overlap_in_words_list(premise, hypothesis)


def get_percentage_count_all(premise, hypothesis):
    if len(premise) == 0:
        return 0
    return 100.0 * get_absolute_count_all(premise, hypothesis) / len(premise)


def get_percentage_count_label(premise_parse, hypothesis_parse, label):
    premise = filter_by_labels(premise_parse, label)
    if len(premise) == 0:
        return 0
    return 100.0 * get_absolute_count_label(
        premise_parse,
        hypothesis_parse, label
    ) / len(premise)


#
# Lexicalized features
#


def get_counts_for_unigrams(sentence):
    counts = defaultdict(int)
    for word in re.findall(r"\w+", sentence):
        counts[word.lower()] += 1
    return counts


def get_counts_for_bigrams(sentence):
    words = re.findall(r"\w+", sentence)
    words = map(lambda x: x.lower(), words)
    counts = defaultdict(int)
    for pair in zip(words, words[1:]):
        counts[pair] += 1
    return counts


def get_cross_unigrams(premise_tags, hypothesis_tags):
    premise_dict = get_POStags_for_sentence(premise_tags)
    hypothesis_dict = get_POStags_for_sentence(hypothesis_tags)

    counts = defaultdict(int)
    for premise_key, premise_value in premise_dict.iteritems():
        for hypothesis_key, hypothesis_value in hypothesis_dict.iteritems():
            if len(list(set(hypothesis_value) & set(premise_value))) > 0:
                counts[(premise_key.lower(), hypothesis_key.lower())] += 1
    return counts


def get_cross_bigrams(premise_tags, hypothesis_tags):
    premise_dict_aux = get_POStags_for_sentence(premise_tags)
    hypothesis_dict_aux = get_POStags_for_sentence(hypothesis_tags)

    premise_dict = {}
    for pair in zip(
        map(lambda x: x.lower(), sorted(premise_dict_aux.keys())),
        map(lambda x: x.lower(), sorted(premise_dict_aux.keys()))[1:]
    ):
        premise_dict[pair] = premise_dict_aux[pair[1]]

    hypothesis_dict = {}
    for pair in zip(
        hypothesis_dict_aux.keys(),
        hypothesis_dict_aux.keys()[1:]
    ):
        hypothesis_dict[pair] = hypothesis_dict_aux[pair[1]]

    counts = defaultdict(int)
    for premise_key, premise_value in premise_dict.iteritems():
        for hypothesis_key, hypothesis_value in hypothesis_dict.iteritems():
            if len(list(set(hypothesis_value) & set(premise_value))) > 0:
                counts[(premise_key, hypothesis_key)] += 1
    return counts


#
# Build dataframe with features for classification
#


def make_feature_list_for_pair(entry):
    # Get unlexicalized features

    # Bleu score
    score_1 = get_bleu_score(entry['sentence1'], entry['sentence2'], 1)
    score_2 = get_bleu_score(entry['sentence1'], entry['sentence2'], 2)
    score_3 = get_bleu_score(entry['sentence1'], entry['sentence2'], 3)
    score_4 = get_bleu_score(entry['sentence1'], entry['sentence2'], 4)

    # Length difference
    length_difference = get_length_difference(
        entry['sentence1'],
        entry['sentence2']
    )

    # Counts
    abs_count_all = get_absolute_count_all(
        re.findall(r"\w+", entry['sentence1']),
        re.findall(r"\w+", entry['sentence2'])
    )
    perc_count_all = get_percentage_count_all(
        re.findall(r"\w+", entry['sentence1']),
        re.findall(r"\w+", entry['sentence2'])
    )

    abs_count_nouns = get_absolute_count_label(
        get_POStags_for_sentence(entry['sentence1_parse']),
        get_POStags_for_sentence(entry['sentence2_parse']),
        NOUN
    )
    perc_count_nouns = get_percentage_count_label(
        get_POStags_for_sentence(entry['sentence1_parse']),
        get_POStags_for_sentence(entry['sentence2_parse']),
        NOUN
    )

    abs_count_verb = get_absolute_count_label(
        get_POStags_for_sentence(entry['sentence1_parse']),
        get_POStags_for_sentence(entry['sentence2_parse']),
        VERB
    )
    perc_count_verb = get_percentage_count_label(
        get_POStags_for_sentence(entry['sentence1_parse']),
        get_POStags_for_sentence(entry['sentence2_parse']),
        VERB
    )

    abs_count_adv = get_absolute_count_label(
        get_POStags_for_sentence(entry['sentence1_parse']),
        get_POStags_for_sentence(entry['sentence2_parse']),
        ADVERB
    )
    perc_count_adv = get_percentage_count_label(
        get_POStags_for_sentence(entry['sentence1_parse']),
        get_POStags_for_sentence(entry['sentence2_parse']),
        ADVERB
    )

    abs_count_adj = get_absolute_count_label(
        get_POStags_for_sentence(entry['sentence1_parse']),
        get_POStags_for_sentence(entry['sentence2_parse']),
        ADJECTIVE
    )
    perc_count_adj = get_percentage_count_label(
        get_POStags_for_sentence(entry['sentence1_parse']),
        get_POStags_for_sentence(entry['sentence2_parse']),
        ADJECTIVE
    )

    # Get lexicalized features

    # Unigram counts
    unigram_counts = get_counts_for_unigrams(entry['sentence1'])

    # Bigram counts
    bigram_counts = get_counts_for_bigrams(entry['sentence1'])

    # Cross unigrams
    cross_unigrams_counts = get_cross_unigrams(
        entry['sentence1_parse'],
        entry['sentence2_parse']
    )

    # Cross bigrams
    cross_bigrams_counts = get_cross_bigrams(
        entry['sentence1_parse'],
        entry['sentence2_parse']
    )

    return pd.Series({
        'gold_label': entry['gold_label'],
        'bleu_score_1': score_1,
        'bleu_score_2': score_2,
        'bleu_score_3': score_3,
        'bleu_score_4': score_4,
        'length_difference': length_difference,
        'absolute_count_all': abs_count_all,
        'absolute_count_nouns': abs_count_nouns,
        'absolute_count_verbs': abs_count_verb,
        'absolute_count_adverbs': abs_count_adv,
        'absolute_count_adjectives': abs_count_adj,
        'percentage_count_all': perc_count_all,
        'percentage_count_nouns': perc_count_nouns,
        'percentage_count_verbs': perc_count_verb,
        'percentage_count_adverbs': perc_count_adv,
        'percentage_count_adjectives': perc_count_adj,
        'unigrams': unigram_counts,
        'bigrams': bigram_counts,
        'cross_unigrams': cross_unigrams_counts,
        'cross_bigrams': cross_bigrams_counts
    })


def mark_idicator_functions_unigrams(entry):
    for word in entry['unigrams'].keys():
        if word in entry:
            entry[word] = 1
    return entry


def mark_idicator_functions_bigrams(entry):
    for word in entry['bigrams'].keys():
        if word in entry:
            entry[word] = 1
    return entry


def mark_idicator_functions_cross_unigrams(entry):
    for word in entry['cross_unigrams'].keys():
        if word in entry:
            entry[word] = 1
    return entry


def mark_idicator_functions_cross_bigrams(entry):
    for word in entry['cross_bigrams'].keys():
        if word in entry:
            entry[word] = 1
    return entry


def make_feature_dataframe(df):
    df_features = df.apply(make_feature_list_for_pair, axis=1)

    # Normalize real valued features
    fields = [
        'bleu_score_1',
        'bleu_score_2',
        'bleu_score_3',
        'bleu_score_4',
        'length_difference',
        'absolute_count_all',
        'absolute_count_nouns',
        'absolute_count_verbs',
        'absolute_count_adverbs',
        'absolute_count_adjectives',
        'percentage_count_all',
        'percentage_count_nouns',
        'percentage_count_verbs',
        'percentage_count_adverbs',
        'percentage_count_adjectives'
    ]
    df_features[fields] = df_features[fields].apply(
        lambda x: (x - x.mean()) / (x.max() - x.min() + 1)
    )

    return df_features


def make_feature_dataframe_extended(df_train, df):
    fields = [
        'unigrams',
        'bigrams',
        'cross_unigrams',
        'cross_bigrams'
    ]

    for field in FIELDS_TO_USE:
        sum_dict = sum_dataframe_dict_no_total(df_train, field)
        keys = sum_dict.keys()
        keys.sort()
        for key in keys:
            df[key] = 0

    df_features = df

    if 'unigrams' in FIELDS_TO_USE:
        df_features = df.apply(mark_idicator_functions_unigrams, axis=1)
    if 'bigrams' in FIELDS_TO_USE:
        df_features = df_features.apply(
            mark_idicator_functions_bigrams,
            axis=1
        )
    if 'cross_unigrams' in FIELDS_TO_USE:
        df_features = df_features.apply(
            mark_idicator_functions_cross_unigrams,
            axis=1
        )
    if 'cross_bigrams' in FIELDS_TO_USE:
        df_features = df_features.apply(
            mark_idicator_functions_cross_bigrams,
            axis=1
        )

    for field in fields:
        df_features = df_features.drop(field, 1)

    # Normalize real valued features
    fields = [
        'bleu_score_1',
        'bleu_score_2',
        'bleu_score_3',
        'bleu_score_4',
        'length_difference',
        'absolute_count_all',
        'absolute_count_nouns',
        'absolute_count_verbs',
        'absolute_count_adverbs',
        'absolute_count_adjectives',
        'percentage_count_all',
        'percentage_count_nouns',
        'percentage_count_verbs',
        'percentage_count_adverbs',
        'percentage_count_adjectives'
    ]
    df_features[fields] = df_features[fields].apply(
        lambda x: (x - x.mean()) / (x.max() - x.min() + 1)
    )

    return df_features


#
# Classification
#


def summarize_dataframe_per_feature(df):
    sum_unigrams_dict = sum_dataframe_dict(df, 'unigrams')

    sum_bigrams_dict = sum_dataframe_dict(df, 'bigrams')

    sum_cross_unigrams_dict = sum_dataframe_dict(df, 'cross_unigrams')

    sum_cross_bigrams_dict = sum_dataframe_dict(df, 'cross_bigrams')

    return {
        'bleu_score_1': df['bleu_score_1'].mean(),
        'bleu_score_2': df['bleu_score_2'].mean(),
        'bleu_score_3': df['bleu_score_3'].mean(),
        'bleu_score_4': df['bleu_score_4'].mean(),
        'length_difference': df['length_difference'].mean(),
        'absolute_count_all': df['absolute_count_all'].mean(),
        'absolute_count_nouns': df['absolute_count_nouns'].mean(),
        'absolute_count_verbs': df['absolute_count_verbs'].mean(),
        'absolute_count_adverbs': df['absolute_count_adverbs'].mean(),
        'absolute_count_adjectives': df['absolute_count_adjectives'].mean(),
        'percentage_count_all': df['percentage_count_all'].mean(),
        'percentage_count_nouns': df['percentage_count_nouns'].mean(),
        'percentage_count_verbs': df['percentage_count_verbs'].mean(),
        'percentage_count_adverbs': df['percentage_count_adverbs'].mean(),
        'percentage_count_adjectives': df['percentage_count_adjectives'].mean(),
        'unigrams': sum_unigrams_dict,
        'bigrams': sum_bigrams_dict,
        'cross_unigrams': sum_cross_unigrams_dict,
        'cross_bigrams': sum_cross_bigrams_dict
    }


def field_difference(field, entry, summary):
    return summary[field] - entry[field]


def compute_distance_between_entry_and_summary(entry, summary):
    fields = [
        'bleu_score_1',
        'bleu_score_2',
        'bleu_score_3',
        'bleu_score_4',
        'length_difference',
        'absolute_count_all',
        'absolute_count_nouns',
        'absolute_count_verbs',
        'absolute_count_adverbs',
        'absolute_count_adjectives',
        'percentage_count_all',
        'percentage_count_nouns',
        'percentage_count_verbs',
        'percentage_count_adverbs',
        'percentage_count_adjectives'
    ]
    distance = 0.0
    for field in fields:
        distance += field_difference(field, entry, summary)

    for field in FIELDS_TO_USE:
        if summary[field]['total_count'] > 0:
            distance += total_dif_first_dict_then_second(
                entry[field],
                summary[field]
            ) / summary[field]['total_count']

    return distance


def get_class(entry, classes):
    min_distance = float('inf')
    result_class = -1

    index = 0
    for cls in classes:
        dist = compute_distance_between_entry_and_summary(
            entry,
            cls
        )
        if dist < min_distance:
            min_distance = dist
            result_class = index
        index += 1

    return result_class


def classify_and_compute_accuracy_simple(train_df, test_df):
    labels = ['contradiction', 'neutral', '-', 'entailment']

    # Split the train dataset for simpler prediction in the classifier
    train_contradictions = get_dataframe_subset_for_label(
        train_df,
        'contradiction'
    )
    train_independents = get_dataframe_subset_for_label(train_df, 'neutral')
    train_undecideds = get_dataframe_subset_for_label(train_df, '-')
    train_entailments = get_dataframe_subset_for_label(train_df, 'entailment')

    train_set = [
        train_contradictions,
        train_independents,
        train_undecideds,
        train_entailments
    ]

    # Split the test dataset for simpler computation of accuracy
    test_contradictions = get_dataframe_subset_for_label(
        test_df,
        'contradiction'
    )
    test_independents = get_dataframe_subset_for_label(test_df, 'neutral')
    test_undecideds = get_dataframe_subset_for_label(test_df, '-')
    test_entailments = get_dataframe_subset_for_label(test_df, 'entailment')

    test_set = [
        test_contradictions,
        test_independents,
        test_undecideds,
        test_entailments
    ]

    # Summarize the classes
    train_contradictions_summary = summarize_dataframe_per_feature(
        train_contradictions
    )
    train_independents_summary = summarize_dataframe_per_feature(
        train_independents
    )
    train_undecideds_summary = summarize_dataframe_per_feature(
        train_undecideds
    )
    train_entailments_summary = summarize_dataframe_per_feature(
        train_entailments
    )

    print('Testing for training set contradictions', file=FILE)
    index_label = 0
    total_count = 0
    for train_set_elem in train_set:
        count = 0
        print('\tFor ' + labels[index_label], file=FILE)
        for index, row in train_set_elem.iterrows():
            label = labels[get_class(
                row,
                [
                    train_contradictions_summary,
                    train_independents_summary,
                    train_undecideds_summary,
                    train_entailments_summary
                ]
            )]
            # print label
            if label == row['gold_label']:
                count += 1
        total_count += count
        if train_set_elem.shape[0] == 0:
            print(100, file=FILE)
        else:
            print(count * 100.0 / train_set_elem.shape[0], file=FILE)
        index_label += 1
    print(
        'Total accuracy: ' + str(total_count * 100.0 / train_df.shape[0]),
        file=FILE
    )

    print('\n', file=FILE)

    print('Testing for test set contradictions', file=FILE)
    index_label = 0
    total_count = 0
    for test_set_elem in test_set:
        count = 0
        print('\tFor ' + labels[index_label], file=FILE)
        for index, row in test_set_elem.iterrows():
            label = labels[get_class(
                row,
                [
                    train_contradictions_summary,
                    train_independents_summary,
                    train_undecideds_summary,
                    train_entailments_summary
                ]
            )]
            if label == row['gold_label']:
                count += 1
        total_count += count
        if test_set_elem.shape[0] == 0:
            print(100, file=FILE)
        else:
            print(count * 100.0 / test_set_elem.shape[0], file=FILE)
        index_label += 1
    print(
        'Total accuracy: ' + str(total_count * 100.0 / test_df.shape[0]),
        file=FILE
    )


def classify_and_compute_accuracy_svm(train_df, test_df):
    df_features_train = make_feature_dataframe_extended(train_df, train_df)
    df_features_test = make_feature_dataframe_extended(train_df, test_df)

    # Create SVM classifier
    clf = svm.SVC()

    # Convert the targets to float
    labels = {
        'contradiction': 1,
        'neutral': 2,
        '-': 3,
        'entailment': 4
    }
    df_features_train['gold_label'] = df_features_train['gold_label'].apply(
        lambda x: labels[x]
    )
    df_features_test['gold_label'] = df_features_test['gold_label'].apply(
        lambda x: labels[x]
    )

    # Train it on the data
    y_train = df_features_train['gold_label'].values
    df_features_train.drop('gold_label', axis=1)
    X_train = df_features_train.values
    clf.fit(X_train, y_train)

    print(clf, file=FILE)

    # Start predicting
    y_test = df_features_test['gold_label'].values
    df_features_test.drop('gold_label', axis=1)
    X_test = df_features_test.values
    results = clf.predict(df_features_test)

    # Compare results and compute accuracy
    print(results, file=FILE)
    print(y_test, file=FILE)
    print(clf.score(X_test, y_test) * 100.0, file=FILE)


if __name__ == '__main__':
    # Parse command line arguments
    try:
        opts, args = getopt.getopt(
            sys.argv[1:],
            'm:',
            [
                'ni=',
                'no=',
                'print_garbage',
                'use_file',
                'use_unigrams',
                'use_all_lexical'
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

    method = 'svm'
    print_garbage = False
    use_file = True
    ni = 100
    no = 1000
    file_extension = 'no_lexical'

    for opt, arg in opts:
        if opt == '-m':
            method = arg
        elif opt == '--ni':
            ni = int(arg)
        elif opt == '--no':
            no = int(arg)
        elif opt == '--print_garbage':
            print_garbage = True
        elif opt == '--use_file':
            use_file = True
        elif opt == '--use_unigrams':
            FIELDS_TO_USE = ['unigrams']
            file_extension = 'only_unigrams'
        elif opt == '--use_all_lexical':
            FIELDS_TO_USE = [
                'unigrams',
                'bigrams',
                'cross_unigrams',
                'cross_bigrams'
            ]
            file_extension = 'all_lexical'

    # Open output file
    if use_file:
        FILE = open(
            '../results/' + time.strftime("%d_%m_%Y_%H_%M_%S") +
            '_' + method + '_ni_' + str(ni) + '_no_' + str(no) + '_' +
            file_extension + '.txt',
            'w'
        )

    # Do the magic
    # TODO: change to train set once everything is developed. Dev set has only
    # a tiny amount of pairs.
    df_train = get_dataframe_from_csv(FULL_CSV_PATH_DEV)
    df_test = get_dataframe_from_csv(FULL_CSV_PATH_TEST)

    if print_garbage:
        print('POS tags for sentence test', file=FILE)
        tagged_words = get_POStags_for_sentence(
            df_test['sentence1_parse'][1]
        )
        print(tagged_words, file=FILE)
        print('\n', file=FILE)

        print('Counts for unigrams', file=FILE)
        counts = get_counts_for_unigrams(
            df_test['sentence1'][1]
        )
        print(counts, file=FILE)
        print('\n', file=FILE)

        print('Counts for bigrams', file=FILE)
        counts_bi = get_counts_for_bigrams(
            df_test['sentence1'][1]
        )
        print(counts_bi, file=FILE)
        print('\n', file=FILE)

        print('Overlap counts', file=FILE)
        print(re.findall(r"\w+", df_test['sentence1'][1]), file=FILE)
        print(df_test['sentence2'][1], file=FILE)
        print(
            get_absolute_count_all(
                re.findall(r"\w+", df_test['sentence1'][1]),
                re.findall(r"\w+", df_test['sentence2'][1])
            ),
            file=FILE
        )
        print(
            get_percentage_count_all(
                re.findall(r"\w+", df_test['sentence1'][1]),
                re.findall(r"\w+", df_test['sentence2'][1])
            ),
            file=FILE
        )
        print('\n', file=FILE)

        print('Filter by labels', file=FILE)
        print(
            filter_by_labels(
                get_POStags_for_sentence(df_test['sentence1_parse'][1]),
                NOUN
            ),
            file=FILE
        )
        print(
            filter_by_labels(
                get_POStags_for_sentence(df_test['sentence2_parse'][1]),
                NOUN
            ),
            file=FILE
        )
        print('\n', file=FILE)

        print('Overlap counts filter', file=FILE)
        print(
            get_absolute_count_label(
                get_POStags_for_sentence(df_test['sentence1_parse'][1]),
                get_POStags_for_sentence(df_test['sentence2_parse'][1]),
                NOUN
            ),
            file=FILE
        )
        print(
            get_percentage_count_label(
                get_POStags_for_sentence(df_test['sentence1_parse'][1]),
                get_POStags_for_sentence(df_test['sentence2_parse'][1]),
                NOUN
            ),
            file=FILE
        )
        print('\n', FILE)

        print('Bleu score', file=FILE)
        print(df_test['sentence1'][1], file=FILE)
        print(df_test['sentence2'][1], file=FILE)
        print(
            get_bleu_score(
                df_test['sentence1'][1],
                df_test['sentence2'][1],
                1
            ),
            file=FILE
        )
        print(
            get_bleu_score(
                df_test['sentence1'][1],
                df_test['sentence2'][1],
                2
            ),
            file=FILE
        )
        print(
            get_bleu_score(
                df_test['sentence1'][1],
                df_test['sentence2'][1],
                3
            ),
            file=FILE
        )
        print(
            get_bleu_score(
                df_test['sentence1'][1],
                df_test['sentence2'][1],
                4
            ),
            file=FILE
        )
        print('\n', file=FILE)

        print('Cross unigrams', file=FILE)
        print(
            get_cross_unigrams(
                df_test['sentence1_parse'][1],
                df_test['sentence2_parse'][1]
            ),
            file=FILE
        )
        print('\n', file=FILE)

        print('Cross bigrams', file=FILE)
        print(
            get_cross_bigrams(
                df_test['sentence1_parse'][1],
                df_test['sentence2_parse'][1]
            ),
            file=FILE
        )
        print('\n', file=FILE)

        print('Checking all features get created', file=FILE)
        print(df_test.xs(327), file=FILE)
        print(make_feature_list_for_pair(df_test.xs(327)), file=FILE)
        print('\n', file=FILE)

        df_features_train = make_feature_dataframe(df_train)
        df_features_test = make_feature_dataframe(df_test)

        print(df_features_train.shape, file=FILE)
        print(df_features_test.shape, file=FILE)

        print(df_features_train[:][:1], file=FILE)

        # print 'Test mean values production'
        # train_contradictions_summary = summarize_dataframe_per_feature(
        #    train_contradictions
        # )
        # train_independent_summary = summarize_dataframe_per_feature(
        #    train_independents
        # )
        # print '\n'

        print(df_train.shape, file=FILE)
        print(df_test.shape, file=FILE)

    df_features_train = make_feature_dataframe(df_train[:][:ni])
    df_features_test = make_feature_dataframe(df_test[:][:no])

    if method == 'classic':
        classify_and_compute_accuracy_simple(
            df_features_train,
            df_features_test
        )
    elif method == 'svm':
        classify_and_compute_accuracy_svm(
            df_features_train[:][:ni],
            df_features_test[:][:no]
        )

    # Close output file
    FILE.close()
