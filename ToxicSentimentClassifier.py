"""
Reference articles for this script:
* https://www.analyticsvidhya.com/blog/2018/02/natural-language-processing-for-beginners-using-textblob/
* https://www.geeksforgeeks.org/twitter-sentiment-analysis-using-python/
* https://stackoverflow.com/questions/24431449/how-to-save-the-result-of-classifier-textblob-naivebayesclassifier
* https://pythonprogramming.net/pickle-classifier-save-nltk-tutorial/
* https://zablo.net/blog/post/twitter-sentiment-analysis-python-scikit-word2vec-nltk-xgboost/
"""

import csv
from textblob import classifiers

max_rows = 5000  # 0 means unlimited

file_encoding = 'utf-8'
file_eol_char = '\n'
rows_to_flush = 10000

train_file_path = 'F:/Kaggle/ToxicCommentClassificationChallenge/train.csv'
test_file_path = 'F:/Kaggle/ToxicCommentClassificationChallenge/test.csv'

toxic_file_path = 'F:/Kaggle/ToxicCommentClassificationChallenge/train_toxic.csv'
severe_toxic_file_path = 'F:/Kaggle/ToxicCommentClassificationChallenge/train_severe_toxic.csv'
obscene_file_path = 'F:/Kaggle/ToxicCommentClassificationChallenge/train_obscene.csv'
threat_file_path = 'F:/Kaggle/ToxicCommentClassificationChallenge/train_threat.csv'
insult_file_path = 'F:/Kaggle/ToxicCommentClassificationChallenge/train_insult.csv'
identity_hate_file_path = 'F:/Kaggle/ToxicCommentClassificationChallenge/train_identity_hate.csv'

source_field_names = [
    "id",
    "comment_text",
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate"
]

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

toxic_classifier = None

rows_read = 0
with open(toxic_file_path,
          'r',
          newline='',
          encoding=file_encoding) as toxic_file:

    toxic_reader = csv.DictReader(toxic_file,
                                  fieldnames=trainer_field_names,
                                  dialect='this_projects_dialect')

    rows = []
    for row in toxic_reader:
        rows_read += 1
        rows.append((row['comment_text'], row['sentiment']))
        if rows_read % rows_to_flush == 0:
            print(f'toxic rows_read: {rows_read:,}')
        if max_rows > 0 and rows_read >= max_rows:
            break

    print(f'toxic rows_read: {rows_read:,}')

    # toxic_classifier = classifiers.NaiveBayesClassifier(rows)
    toxic_classifier = classifiers.DecisionTreeClassifier(rows)

    print(f'toxic classifier trained!')
