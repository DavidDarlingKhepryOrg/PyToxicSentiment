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

from textblob import TextBlob

max_rows_read = 100  # 0 means unlimited

file_encoding = 'utf-8'
file_eol_char = '\n'
rows_per_batch = 100

classifier_type = 'NaiveBayes'

src_folder = 'F:/Kaggle/ToxicCommentClassificationChallenge'
tgt_folder = 'F:/Kaggle/ToxicCommentClassificationChallenge'

pickled_files = {
    'toxic': f'{src_folder}/Classifier_{classifier_type}_toxic.pickle',
    'severe_toxic': f'{src_folder}/Classifier_{classifier_type}_severe_toxic.pickle',
    'obscene': f'{src_folder}/Classifier_{classifier_type}_obscene.pickle',
    'threat': f'{src_folder}/Classifier_{classifier_type}_threat.pickle',
    'insult': f'{src_folder}/Classifier_{classifier_type}_insult.pickle',
    'identity_hate': f'{src_folder}/Classifier_{classifier_type}_identity_hate.pickle',
}

source_files = {
    'toxic': f'{src_folder}/accuracy_toxic.csv',
    'severe_toxic': f'{src_folder}/accuracy_severe_toxic.csv',
    'obscene': f'{src_folder}/accuracy_obscene.csv',
    'threat': f'{src_folder}/accuracy_threat.csv',
    'insult': f'{src_folder}/accuracy_insult.csv',
    'identity_hate': f'{src_folder}/accuracy_identity_hate.csv',
}

csv.register_dialect('this_projects_dialect',
                     delimiter=',',
                     quotechar='"',
                     quoting=csv.QUOTE_MINIMAL,
                     lineterminator=file_eol_char)

for key, source_file_path in source_files.items():

    print(f'key: {key}, file: {source_file_path}')

    rows_read = 0
    rows_used = 0

    # load the pickled classifier
    # from the specified file path
    pickle_file_path = pickled_files[key]
    # load the classifier from the pickled file
    with open(pickle_file_path, 'rb') as pickle_file:
        print(f'Loading of {key} classifier from pickle file: {pickle_file_path} begun')
        classifier = pickle.load(pickle_file, fix_imports=False, encoding=file_encoding)
        print(f'Loading of {key} classifier from pickle file: {pickle_file_path} ended')

        # instantiate the source file's reader
        with open(source_file_path,
                  'r',
                  newline='',
                  encoding=file_encoding) as accuracy_file:

            csv_reader = csv.reader(accuracy_file,
                                    dialect='this_projects_dialect')
            rows = []
            print(f'Loading of {key} accuracy file rows begun')
            rows_read = 0
            for row in csv_reader:
                rows_read += 1
                rows.append(row)
                if rows_read >= max_rows_read > 0:
                    break
            print(f'Loading of {key} accuracy file rows ended')
            print(classifier.accuracy(rows))
            classifier.show_informative_features(10)

            print('========================================================')
