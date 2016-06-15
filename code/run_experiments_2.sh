#!/bin/bash

for niv in {100000, 150000, 200000, 300000}
do
        for nov in {10000}
        do
                echo 'Computing things for ' $niv ' and ' $nov
                time python feature_based_contradiction_detection.py -m svm --use_full_train --ni=$niv --no-$nov --use_file
                time python feature_based_contradiction_detection.py -m svm --use_full_train --ni=$niv --no=$nov --use_file --use_unigrams
                time python feature_based_contradiction_detection.py -m svm --use_full_train --ni=$niv --no=$nov --use_file --use_all_lexical
        done
done

for niv in {100000, 150000, 200000, 300000}
do
        for nov in {10000}
        do
                echo 'Computing things for ' $niv ' and ' $nov
                time python feature_based_contradiction_detection.py -m randomforest --use_full_train --ni=$niv --no-$nov --use_file
                time python feature_based_contradiction_detection.py -m randomforest --use_full_train --ni=$niv --no=$nov --use_file --use_unigrams
                time python feature_based_contradiction_detection.py -m randomforest --use_full_train --ni=$niv --no=$nov --use_file --use_all_lexical
        done
done

