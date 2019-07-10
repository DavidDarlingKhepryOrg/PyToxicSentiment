"""
Reference articles for this script:
* https://www.analyticsvidhya.com/blog/2018/02/natural-language-processing-for-beginners-using-textblob/
* https://www.geeksforgeeks.org/twitter-sentiment-analysis-using-python/
* https://stackoverflow.com/questions/24431449/how-to-save-the-result-of-classifier-textblob-naivebayesclassifier
* https://pythonprogramming.net/pickle-classifier-save-nltk-tutorial/
* https://zablo.net/blog/post/twitter-sentiment-analysis-python-scikit-word2vec-nltk-xgboost/
"""

import csv
import pickle
import sys

from textblob import classifiers

max_rows_used = 1000  # 0 means unlimited

file_encoding = 'utf-8'
file_eol_char = '\n'
rows_per_batch = 100

classifier_type = 'NaiveBayes'

src_folder = 'F:/Kaggle/ToxicCommentClassificationChallenge'
tgt_folder = 'F:/Kaggle/ToxicCommentClassificationChallenge'

training_files = {
    'toxic': f'{src_folder}/train_toxic.csv',
    'severe_toxic': f'{src_folder}/train_severe_toxic.csv',
    'obscene': f'{src_folder}/train_obscene.csv',
    'threat': f'{src_folder}/train_threat.csv',
    'insult': f'{src_folder}/train_insult.csv',
    'identity_hate': f'{src_folder}/train_identity_hate.csv',
}

trainer_field_names = [
    "id",
    "comment_text",
    "sentiment"
]

csv.register_dialect('this_projects_dialect',
                     delimiter=',',
                     quotechar='"',
                     quoting=csv.QUOTE_MINIMAL,
                     lineterminator=file_eol_char)

for key, training_file_path in training_files.items():

    classifier = None
    rows_read = 0
    rows_used = 0
    with open(training_file_path,
              'r',
              newline='',
              encoding=file_encoding) as training_file:

        csv_dict_reader = csv.DictReader(training_file,
                                         fieldnames=trainer_field_names,
                                         dialect='this_projects_dialect')

        first_batch = True
        rows = []
        for row in csv_dict_reader:
            rows_read += 1
            if row['sentiment'] == 'neg':
                rows_used += 1
                rows.append((row['comment_text'], row['sentiment']))
                if rows_used % rows_per_batch == 0:
                    if first_batch:
                        if classifier_type == 'NaiveBayes':
                            classifier = classifiers.NaiveBayesClassifier(rows)
                        elif classifier_type == 'DecisionTree':
                            classifier = classifiers.DecisionTreeClassifier(rows)
                        else:
                            print(f'Invalid classifier type specified: {classifier_type}')
                            sys.exit(1)
                        first_batch = False
                    else:
                        classifier.update(rows)
                    rows.clear()
                    print(f'{key} {classifier_type} classifier updated, rows_read: {rows_read:,}, rows_used: {rows_used:,}')

            if max_rows_used > 0 and rows_used >= max_rows_used:
                break

        print(f'{key} {classifier_type} classifier updated, rows_read: {rows_read:,}, rows_used: {rows_used:,}')

    print(f'Pickling of {key} {classifier_type} classifier started')
    pickle_file_path = f'{tgt_folder}/Classifier_{classifier_type}_{key}.pickle'
    with open(pickle_file_path, 'wb') as pickle_file:
        pickle.dump(classifier, pickle_file, protocol=pickle.HIGHEST_PROTOCOL, fix_imports=False)
    print(f'Pickling of {key} {classifier_type} classifier finished')
    print('========================================================')

