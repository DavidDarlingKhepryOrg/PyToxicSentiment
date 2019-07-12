"""
ToxicSentimentSplitter.py

Reference articles for this script:
* https://www.analyticsvidhya.com/blog/2018/02/natural-language-processing-for-beginners-using-textblob/
* https://www.geeksforgeeks.org/twitter-sentiment-analysis-using-python/
"""

import csv
import re
from nltk.corpus import stopwords
from textblob import TextBlob

max_rows = 0  # 0 means unlimited

file_encoding = 'utf-8'
file_eol_char = '\n'
rows_to_flush = 1000

source_file_path = 'F:/Kaggle/ToxicCommentClassificationChallenge/train.csv'

train_toxic_file_path = 'F:/Kaggle/ToxicCommentClassificationChallenge/train_toxic.csv'
train_severe_toxic_file_path = 'F:/Kaggle/ToxicCommentClassificationChallenge/train_severe_toxic.csv'
train_obscene_file_path = 'F:/Kaggle/ToxicCommentClassificationChallenge/train_obscene.csv'
train_threat_file_path = 'F:/Kaggle/ToxicCommentClassificationChallenge/train_threat.csv'
train_insult_file_path = 'F:/Kaggle/ToxicCommentClassificationChallenge/train_insult.csv'
train_identity_hate_file_path = 'F:/Kaggle/ToxicCommentClassificationChallenge/train_identity_hate.csv'

sentence_toxic_file_path = 'F:/Kaggle/ToxicCommentClassificationChallenge/sentence_toxic.csv'
sentence_severe_toxic_file_path = 'F:/Kaggle/ToxicCommentClassificationChallenge/sentence_severe_toxic.csv'
sentence_obscene_file_path = 'F:/Kaggle/ToxicCommentClassificationChallenge/sentence_obscene.csv'
sentence_threat_file_path = 'F:/Kaggle/ToxicCommentClassificationChallenge/sentence_threat.csv'
sentence_insult_file_path = 'F:/Kaggle/ToxicCommentClassificationChallenge/sentence_insult.csv'
sentence_identity_hate_file_path = 'F:/Kaggle/ToxicCommentClassificationChallenge/sentence_identity_hate.csv'

accuracy_toxic_file_path = 'F:/Kaggle/ToxicCommentClassificationChallenge/accuracy_toxic.csv'
accuracy_severe_toxic_file_path = 'F:/Kaggle/ToxicCommentClassificationChallenge/accuracy_severe_toxic.csv'
accuracy_obscene_file_path = 'F:/Kaggle/ToxicCommentClassificationChallenge/accuracy_obscene.csv'
accuracy_threat_file_path = 'F:/Kaggle/ToxicCommentClassificationChallenge/accuracy_threat.csv'
accuracy_insult_file_path = 'F:/Kaggle/ToxicCommentClassificationChallenge/accuracy_insult.csv'
accuracy_identity_hate_file_path = 'F:/Kaggle/ToxicCommentClassificationChallenge/accuracy_identity_hate.csv'

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

target_field_names = [
    "id",
    "comment_text",
    "sentiment"
]

accuracy_field_names = [
    "comment_text",
    "sentiment"
]

pos_neg_dict = {
    '0': 'pos',
    '1': 'neg'
}

regexs = {
    'http': re.compile(r"http.?://[^\s]+[\s]?", re.IGNORECASE),
    'https': re.compile(r"https.?://[^\s]+[\s]?", re.IGNORECASE),
    'numbers': re.compile(r"\s?[0-9]+\.?[0-9]*", re.IGNORECASE)
}

cached_stopwords = stopwords.words("english")
cached_stopwords.append('wiki')


# replace pattern characters with replacement characters
# until the source string has no pattern characters remaining
def replace_chars(src_str, pat_chars='  ', rpl_chars=' '):
    while src_str.find(pat_chars) > -1:
        src_str = src_str.replace(pat_chars, rpl_chars)
    return src_str


# cleanse source string of
# any non-alphanumeric characters,
# allowing single spaces as the exception
def cleanse_text(src_str):
    src_str = src_str.lower()
    for regex_compiled in regexs.values():
        src_str = re.sub(regex_compiled, '', src_str, re.MULTILINE)
    tmp_str_1 = src_str\
        .replace(file_eol_char, ' ')\
        .replace('-', ' ')\
        .replace('_', ' ')\
        .replace('.', ' ')\
        .replace(',', ' ')\
        .replace(';', ' ')\
        .replace("'", '')\
        .replace(':', ' ')
    tmp_str_1 = replace_chars(tmp_str_1, '  ', ' ')
    tmp_str_2 = ''
    for c in tmp_str_1:
        if c.isalnum() or c in [" ", "'"]:
            tmp_str_2 += c
    tmp_str_2 = replace_chars(tmp_str_2, '  ', ' ')
    # remove stop words from text
    tmp_str_2 = ' '.join([word for word in tmp_str_2.split() if word not in cached_stopwords])
    tmp_str_2 = tmp_str_2.strip()
    return tmp_str_2


csv.register_dialect('this_projects_dialect',
                     delimiter=',',
                     quotechar='"',
                     quoting=csv.QUOTE_MINIMAL,
                     lineterminator=file_eol_char)

rows_read = 0
with open(train_toxic_file_path,
          'w',
          newline='',
          encoding=file_encoding) as train_toxic_file:
    train_toxic_writer = csv.DictWriter(train_toxic_file,
                                        fieldnames=target_field_names,
                                        dialect='this_projects_dialect')
    train_toxic_writer.writeheader()
    with open(train_severe_toxic_file_path,
              'w',
              newline='',
              encoding=file_encoding) as train_severe_toxic_file:
        train_severe_toxic_writer = csv.DictWriter(train_severe_toxic_file,
                                                   fieldnames=target_field_names,
                                                   dialect='this_projects_dialect')
        train_severe_toxic_writer.writeheader()
        with open(train_obscene_file_path,
                  'w',
                  newline='',
                  encoding=file_encoding) as train_obscene_file:
            train_obscene_writer = csv.DictWriter(train_obscene_file,
                                                  fieldnames=target_field_names,
                                                  dialect='this_projects_dialect')
            train_obscene_writer.writeheader()
            with open(train_threat_file_path,
                      'w', newline='',
                      encoding=file_encoding) as train_threat_file:
                train_threat_writer = csv.DictWriter(train_threat_file,
                                                     fieldnames=target_field_names,
                                                     dialect='this_projects_dialect')
                train_threat_writer.writeheader()
                with open(train_insult_file_path,
                          'w',
                          newline='',
                          encoding=file_encoding) as train_insult_file:
                    train_insult_writer = csv.DictWriter(train_insult_file,
                                                         fieldnames=target_field_names,
                                                         dialect='this_projects_dialect')
                    train_insult_writer.writeheader()
                    with open(train_identity_hate_file_path,
                              'w',
                              newline='',
                              encoding=file_encoding) as train_identity_hate_file:
                        train_identity_hate_writer = csv.DictWriter(train_identity_hate_file,
                                                                    fieldnames=target_field_names,
                                                                    dialect='this_projects_dialect')
                        train_identity_hate_writer.writeheader()
                        with open(source_file_path,
                                  'r',
                                  newline='',
                                  encoding=file_encoding) as src_file:
                            src_reader = csv.DictReader(src_file,
                                                        delimiter=',',
                                                        quotechar='"',
                                                        quoting=csv.QUOTE_MINIMAL)
                            for src_row in src_reader:
                                tgt_text = src_row['comment_text'].replace(file_eol_char, ' ')
                                # tgt_text = cleanse_text(tgt_text)
                                if tgt_text != '':
                                    tgt_row = dict()
                                    tgt_row['id'] = src_row['id']
                                    tgt_row['comment_text'] = tgt_text
                                    tgt_row['sentiment'] = pos_neg_dict[src_row['toxic']]
                                    train_toxic_writer.writerow(tgt_row)
                                    tgt_row['sentiment'] = pos_neg_dict[src_row['severe_toxic']]
                                    train_severe_toxic_writer.writerow(tgt_row)
                                    tgt_row['sentiment'] = pos_neg_dict[src_row['obscene']]
                                    train_obscene_writer.writerow(tgt_row)
                                    tgt_row['sentiment'] = pos_neg_dict[src_row['threat']]
                                    train_threat_writer.writerow(tgt_row)
                                    tgt_row['sentiment'] = pos_neg_dict[src_row['insult']]
                                    train_insult_writer.writerow(tgt_row)
                                    tgt_row['sentiment'] = pos_neg_dict[src_row['identity_hate']]
                                    train_identity_hate_writer.writerow(tgt_row)
                                rows_read += 1
                                if rows_read % rows_to_flush == 0:
                                    train_toxic_file.flush()
                                    train_severe_toxic_file.flush()
                                    train_obscene_file.flush()
                                    train_threat_file.flush()
                                    train_insult_file.flush()
                                    train_identity_hate_file.flush()
                                    print(f'rows_read: {rows_read:,}')

                                if rows_read >= max_rows > 0:
                                    break

print(f'rows_read: {rows_read:,}')

rows_read = 0
with open(sentence_toxic_file_path,
          'w',
          newline='',
          encoding=file_encoding) as sentence_toxic_file:
    sentence_toxic_writer = csv.DictWriter(sentence_toxic_file,
                                           fieldnames=target_field_names,
                                           dialect='this_projects_dialect')
    sentence_toxic_writer.writeheader()
    with open(sentence_severe_toxic_file_path,
              'w',
              newline='',
              encoding=file_encoding) as sentence_severe_toxic_file:
        sentence_severe_toxic_writer = csv.DictWriter(sentence_severe_toxic_file,
                                                      fieldnames=target_field_names,
                                                      dialect='this_projects_dialect')
        sentence_severe_toxic_writer.writeheader()
        with open(sentence_obscene_file_path,
                  'w',
                  newline='',
                  encoding=file_encoding) as sentence_obscene_file:
            sentence_obscene_writer = csv.DictWriter(sentence_obscene_file,
                                                     fieldnames=target_field_names,
                                                     dialect='this_projects_dialect')
            sentence_obscene_writer.writeheader()
            with open(sentence_threat_file_path,
                      'w', newline='',
                      encoding=file_encoding) as sentence_threat_file:
                sentence_threat_writer = csv.DictWriter(sentence_threat_file,
                                                        fieldnames=target_field_names,
                                                        dialect='this_projects_dialect')
                sentence_threat_writer.writeheader()
                with open(sentence_insult_file_path,
                          'w',
                          newline='',
                          encoding=file_encoding) as sentence_insult_file:
                    sentence_insult_writer = csv.DictWriter(sentence_insult_file,
                                                            fieldnames=target_field_names,
                                                            dialect='this_projects_dialect')
                    sentence_insult_writer.writeheader()
                    with open(sentence_identity_hate_file_path,
                              'w',
                              newline='',
                              encoding=file_encoding) as sentence_identity_hate_file:
                        sentence_identity_hate_writer = csv.DictWriter(sentence_identity_hate_file,
                                                                       fieldnames=target_field_names,
                                                                       dialect='this_projects_dialect')
                        sentence_identity_hate_writer.writeheader()
                        with open(source_file_path,
                                  'r',
                                  newline='',
                                  encoding=file_encoding) as src_file:
                            src_reader = csv.DictReader(src_file,
                                                        delimiter=',',
                                                        quotechar='"',
                                                        quoting=csv.QUOTE_MINIMAL)
                            for src_row in src_reader:
                                tgt_text = src_row['comment_text'].replace(file_eol_char, ' ')
                                text_blob = TextBlob(tgt_text)
                                for sentence in text_blob.sentences:
                                    # tgt_text = cleanse_text(tgt_text)
                                    sentence = sentence.strip()
                                    if sentence != '':
                                        tgt_row = dict()
                                        tgt_row['id'] = src_row['id']
                                        tgt_row['comment_text'] = sentence
                                        tgt_row['sentiment'] = pos_neg_dict[src_row['toxic']]
                                        sentence_toxic_writer.writerow(tgt_row)
                                        tgt_row['sentiment'] = pos_neg_dict[src_row['severe_toxic']]
                                        sentence_severe_toxic_writer.writerow(tgt_row)
                                        tgt_row['sentiment'] = pos_neg_dict[src_row['obscene']]
                                        sentence_obscene_writer.writerow(tgt_row)
                                        tgt_row['sentiment'] = pos_neg_dict[src_row['threat']]
                                        sentence_threat_writer.writerow(tgt_row)
                                        tgt_row['sentiment'] = pos_neg_dict[src_row['insult']]
                                        sentence_insult_writer.writerow(tgt_row)
                                        tgt_row['sentiment'] = pos_neg_dict[src_row['identity_hate']]
                                        sentence_identity_hate_writer.writerow(tgt_row)
                                rows_read += 1
                                if rows_read % rows_to_flush == 0:
                                    sentence_toxic_file.flush()
                                    sentence_severe_toxic_file.flush()
                                    sentence_obscene_file.flush()
                                    sentence_threat_file.flush()
                                    sentence_insult_file.flush()
                                    sentence_identity_hate_file.flush()
                                    print(f'rows_read: {rows_read:,}')

                                if rows_read >= max_rows > 0:
                                    break

print(f'rows_read: {rows_read:,}')

rows_read = 0
with open(accuracy_toxic_file_path,
          'w',
          newline='',
          encoding=file_encoding) as accuracy_toxic_file:
    accuracy_toxic_writer = csv.DictWriter(accuracy_toxic_file,
                                           fieldnames=accuracy_field_names,
                                           dialect='this_projects_dialect')
    # accuracy_toxic_writer.writeheader()
    with open(accuracy_severe_toxic_file_path,
              'w',
              newline='',
              encoding=file_encoding) as accuracy_severe_toxic_file:
        accuracy_severe_toxic_writer = csv.DictWriter(accuracy_severe_toxic_file,
                                                      fieldnames=accuracy_field_names,
                                                      dialect='this_projects_dialect')
        # accuracy_severe_toxic_writer.writeheader()
        with open(accuracy_obscene_file_path,
                  'w',
                  newline='',
                  encoding=file_encoding) as accuracy_obscene_file:
            accuracy_obscene_writer = csv.DictWriter(accuracy_obscene_file,
                                                     fieldnames=accuracy_field_names,
                                                     dialect='this_projects_dialect')
            # accuracy_obscene_writer.writeheader()
            with open(accuracy_threat_file_path,
                      'w', newline='',
                      encoding=file_encoding) as accuracy_threat_file:
                accuracy_threat_writer = csv.DictWriter(accuracy_threat_file,
                                                        fieldnames=accuracy_field_names,
                                                        dialect='this_projects_dialect')
                # accuracy_threat_writer.writeheader()
                with open(accuracy_insult_file_path,
                          'w',
                          newline='',
                          encoding=file_encoding) as accuracy_insult_file:
                    accuracy_insult_writer = csv.DictWriter(accuracy_insult_file,
                                                            fieldnames=accuracy_field_names,
                                                            dialect='this_projects_dialect')
                    # accuracy_insult_writer.writeheader()
                    with open(accuracy_identity_hate_file_path,
                              'w',
                              newline='',
                              encoding=file_encoding) as accuracy_identity_hate_file:
                        accuracy_identity_hate_writer = csv.DictWriter(accuracy_identity_hate_file,
                                                                       fieldnames=accuracy_field_names,
                                                                       dialect='this_projects_dialect')
                        # accuracy_identity_hate_writer.writeheader()
                        with open(source_file_path,
                                  'r',
                                  newline='',
                                  encoding=file_encoding) as src_file:
                            src_reader = csv.DictReader(src_file,
                                                        delimiter=',',
                                                        quotechar='"',
                                                        quoting=csv.QUOTE_MINIMAL)
                            for src_row in src_reader:
                                tgt_text = src_row['comment_text'].replace(file_eol_char, ' ')
                                # tgt_text = cleanse_text(tgt_text)
                                if tgt_text != '':
                                    tgt_row = dict()
                                    tgt_row['comment_text'] = tgt_text
                                    tgt_row['sentiment'] = pos_neg_dict[src_row['toxic']]
                                    accuracy_toxic_writer.writerow(tgt_row)
                                    tgt_row['sentiment'] = pos_neg_dict[src_row['severe_toxic']]
                                    accuracy_severe_toxic_writer.writerow(tgt_row)
                                    tgt_row['sentiment'] = pos_neg_dict[src_row['obscene']]
                                    accuracy_obscene_writer.writerow(tgt_row)
                                    tgt_row['sentiment'] = pos_neg_dict[src_row['threat']]
                                    accuracy_threat_writer.writerow(tgt_row)
                                    tgt_row['sentiment'] = pos_neg_dict[src_row['insult']]
                                    accuracy_insult_writer.writerow(tgt_row)
                                    tgt_row['sentiment'] = pos_neg_dict[src_row['identity_hate']]
                                    accuracy_identity_hate_writer.writerow(tgt_row)
                                rows_read += 1
                                if rows_read % rows_to_flush == 0:
                                    accuracy_toxic_file.flush()
                                    accuracy_severe_toxic_file.flush()
                                    accuracy_obscene_file.flush()
                                    accuracy_threat_file.flush()
                                    accuracy_insult_file.flush()
                                    accuracy_identity_hate_file.flush()
                                    print(f'rows_read: {rows_read:,}')

                                if rows_read >= max_rows > 0:
                                    break

print(f'rows_read: {rows_read:,}')
