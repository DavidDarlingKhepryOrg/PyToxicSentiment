"""
Reference articles for this script:
* https://www.analyticsvidhya.com/blog/2018/02/natural-language-processing-for-beginners-using-textblob/
* https://www.geeksforgeeks.org/twitter-sentiment-analysis-using-python/
"""

import csv

max_rows = 0  # 0 means unlimited

file_encoding = 'utf-8'
file_eol_char = '\n'
rows_to_flush = 10000

source_file_path = 'F:/Kaggle/ToxicCommentClassificationChallenge/train.csv'
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

target_field_names = [
    "id",
    "comment_text",
    "sentiment"
]

pos_neg_dict = {
    '0': 'pos',
    '1': 'neg'
}


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
    tmp_str_1 = src_str\
        .replace(file_eol_char, ' ')\
        .replace('-', ' ')\
        .replace('_', ' ')
    tmp_str_1 = replace_chars(tmp_str_1, '  ', ' ')
    tmp_str_2 = ''
    for c in tmp_str_1:
        if c.isalnum() or c in [" ", "'"]:
            tmp_str_2 += c
    tmp_str_2 = replace_chars(tmp_str_2, '  ', ' ')
    tmp_str_2 = tmp_str_2.strip()
    return tmp_str_2


csv.register_dialect('this_projects_dialect',
                     delimiter=',',
                     quotechar='"',
                     quoting=csv.QUOTE_MINIMAL,
                     lineterminator=file_eol_char)

rows_read = 0
with open(toxic_file_path,
          'w',
          newline='',
          encoding=file_encoding) as toxic_file:
    toxic_writer = csv.DictWriter(toxic_file,
                                  fieldnames=target_field_names,
                                  dialect='this_projects_dialect')
    toxic_writer.writeheader()
    with open(severe_toxic_file_path,
              'w',
              newline='',
              encoding=file_encoding) as severe_toxic_file:
        severe_toxic_writer = csv.DictWriter(severe_toxic_file,
                                             fieldnames=target_field_names,
                                             dialect='this_projects_dialect')
        severe_toxic_writer.writeheader()
        with open(obscene_file_path,
                  'w',
                  newline='',
                  encoding=file_encoding) as obscene_file:
            obscene_writer = csv.DictWriter(obscene_file,
                                            fieldnames=target_field_names,
                                            dialect='this_projects_dialect')
            obscene_writer.writeheader()
            with open(threat_file_path,
                      'w', newline='',
                      encoding=file_encoding) as threat_file:
                threat_writer = csv.DictWriter(threat_file,
                                               fieldnames=target_field_names,
                                               dialect='this_projects_dialect')
                threat_writer.writeheader()
                with open(insult_file_path,
                          'w',
                          newline='',
                          encoding=file_encoding) as insult_file:
                    insult_writer = csv.DictWriter(insult_file,
                                                   fieldnames=target_field_names,
                                                   dialect='this_projects_dialect')
                    insult_writer.writeheader()
                    with open(identity_hate_file_path,
                              'w',
                              newline='',
                              encoding=file_encoding) as identity_hate_file:
                        identity_hate_writer = csv.DictWriter(identity_hate_file,
                                                              fieldnames=target_field_names,
                                                              dialect='this_projects_dialect')
                        identity_hate_writer.writeheader()
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
                                tgt_text = cleanse_text(tgt_text)
                                if tgt_text != '':
                                    tgt_row = {'id': src_row['id'], 'comment_text': tgt_text}
                                    tgt_row['sentiment'] = pos_neg_dict[src_row['toxic']]
                                    toxic_writer.writerow(tgt_row)
                                    tgt_row['sentiment'] = pos_neg_dict[src_row['severe_toxic']]
                                    severe_toxic_writer.writerow(tgt_row)
                                    tgt_row['sentiment'] = pos_neg_dict[src_row['obscene']]
                                    obscene_writer.writerow(tgt_row)
                                    tgt_row['sentiment'] = pos_neg_dict[src_row['threat']]
                                    threat_writer.writerow(tgt_row)
                                    tgt_row['sentiment'] = pos_neg_dict[src_row['insult']]
                                    insult_writer.writerow(tgt_row)
                                    tgt_row['sentiment'] = pos_neg_dict[src_row['identity_hate']]
                                    identity_hate_writer.writerow(tgt_row)
                                rows_read += 1
                                if rows_read % rows_to_flush == 0:
                                    toxic_file.flush()
                                    severe_toxic_file.flush()
                                    obscene_file.flush()
                                    threat_file.flush()
                                    insult_file.flush()
                                    identity_hate_file.flush()
                                    print(f'rows_read: {rows_read:,}')

                                if max_rows > 0 and rows_read >= max_rows:
                                    break

print(f'rows_read: {rows_read:,}')
