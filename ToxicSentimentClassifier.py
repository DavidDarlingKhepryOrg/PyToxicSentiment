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
    'toxic': f'{src_folder}/train_toxic.csv',
    'severe_toxic': f'{src_folder}/train_severe_toxic.csv',
    'obscene': f'{src_folder}/train_obscene.csv',
    'threat': f'{src_folder}/train_threat.csv',
    'insult': f'{src_folder}/train_insult.csv',
    'identity_hate': f'{src_folder}/train_identity_hate.csv',
}

target_files = {
    'toxic': f'{src_folder}/classified_toxic.csv',
    'severe_toxic': f'{src_folder}/classified_severe_toxic.csv',
    'obscene': f'{src_folder}/classified_obscene.csv',
    'threat': f'{src_folder}/classified_threat.csv',
    'insult': f'{src_folder}/classified_insult.csv',
    'identity_hate': f'{src_folder}/classified_identity_hate.csv',
}

source_field_names = [
    "id",
    "sentiment",
    "comment_text"
]

target_field_names = [
    "id",
    "sentiment",
    "comment_text"
]

csv.register_dialect('this_projects_dialect',
                     delimiter=',',
                     quotechar='"',
                     quoting=csv.QUOTE_MINIMAL,
                     lineterminator=file_eol_char)

for key, source_file_path in source_files.items():

    rows_read = 0
    rows_used = 0
    target_file_path = target_files[key]
    with open(target_file_path,
              'w',
              newline='',
              encoding=file_encoding) as target_file:

        # load the pickled classifier
        # from the specified file path
        pickle_file_path = pickled_files[key]
        # load the classifier from the pickled file
        with open(pickle_file_path, 'rb') as pickle_file:
            print(f'Loading of {key} classifier from pickle file: {pickle_file_path} begun')
            classifier = pickle.load(pickle_file, fix_imports=False, encoding=file_encoding)
            print(f'Loading of {key} classifier from pickle file: {pickle_file_path} ended')
            classifier.show_informative_features(100)
            # instantiate the target file's writer
            csv_dict_writer = csv.DictWriter(target_file,
                                             fieldnames=target_field_names,
                                             dialect='this_projects_dialect')

            # instantiate the source file's reader
            with open(source_file_path,
                      'r',
                      newline='',
                      encoding=file_encoding) as training_file:

                csv_dict_reader = csv.DictReader(training_file,
                                                 fieldnames=source_field_names,
                                                 dialect='this_projects_dialect')

                for row in csv_dict_reader:
                    rows_read += 1
                    row['sentiment'] = TextBlob(row['comment_text'], classifier=classifier).classify()
                    csv_dict_writer.writerow(row)
                    if rows_read % rows_per_batch == 0:
                        target_file.flush()
                        print(f'{key} {classifier_type} rows_read: {rows_read:,}')

                    if rows_read >= max_rows_read > 0:
                        break

                print(f'{key} {classifier_type} rows_read: {rows_read:,}')

                print('========================================================')
