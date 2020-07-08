import numpy as np
import pandas as pd
import nltk
import argparse
import json
import os
import re

from sklearn.model_selection import train_test_split


"""Add here incrementally new conditioning hypothesis, increment the final subscript"""
def conditioning_hyp1(df):
    return df['text']

import string as strng

def erase_leading_punctuation(string):
  for i in range(len(string)):
    if not string[i] in strng.punctuation and not string[i]== ' ':
      return string[i:]


def formatter(string):
    string = re.sub(r'^https?:\/\/.*[\r\n]*', '', string, flags=re.MULTILINE)
    lst = string.lower().replace('."', '"').replace('u.s.','us').replace('e.g.','eg').replace('\n', '').split('.')
    inter = "\n".join([" ".join(nltk.word_tokenize(erase_leading_punctuation(s))) for s in lst if s != '' and erase_leading_punctuation(s)!=None])
    return inter

def tokenize_csv(csv_file):
    tokenized_csv = csv_file.replace('.csv', '')+'_tokenized.csv'
    if not os.path.exists(tokenized_csv):
        wikihow_df = pd.read_csv(csv_file)
        wikihow_df = wikihow_df[wikihow_df['text'].isna() == False]
        print(wikihow_df.shape)
        wikihow_df = wikihow_df[wikihow_df['text'].apply(len) > wikihow_df['headline'].apply(len)]
        print(wikihow_df.shape)
        # Change conditioning substructure here by redefining or adding conditioning function
        wikihow_df['conditioning'] = conditioning_hyp1(wikihow_df)
        wikihow_df_c1 = wikihow_df.drop(['title', 'text'], axis=1)
        wikihow_df_c1.rename(columns={'headline': 'summaries', 'conditioning': 'doc'}, inplace=True)

        nltk.download('punkt')

        # Remove empty lines
        wikihow_df_c1 = wikihow_df_c1.applymap(lambda x: formatter(x))

        wikihow_df_c1.to_csv(tokenized_csv)
        print("Successfully finished tokenizing {} to {}.\n".format(
            csv_file, tokenized_csv))
    else:
        print("Tokenized csv already present")
        pass


def csv_to_json(csv_file, single):
    wikihow_df = pd.read_csv(csv_file)

    train_df, test_df = train_test_split(wikihow_df, test_size=0.05)
    test_df, val_df = train_test_split(test_df, test_size=0.5)

    for df in [train_df, test_df, val_df]:
        articles = df['doc'].to_numpy()
        summaries = df['summaries'].to_numpy()

        dir = 'data/'
        if df.equals(train_df):
            out_file = 'train'
        elif df.equals(test_df):
            out_file = 'test'
        else:
            out_file = 'val'

        with open(dir+out_file + ".json", 'w') as f:
            articles_n = articles.size
            examples = []
            for idx in range(articles_n):
                if idx % 1000 == 0:
                    print("Writing story {} of {}; {:.2f} percent done".format(
                        idx, articles_n, float(idx) * 100.0 / float(articles_n)))

                # Get the strings to write to .bin file
                article_sents = articles[idx]
                summary_sents = summaries[idx]

                # Write to JSON file
                js_example = {}
                js_example['id'] = "{}".format(idx)
                js_example['doc'] = article_sents
                js_example['summaries'] = summary_sents
                js_example['labels'] = "0\n0\n0\n0" #placeholder (Not needed at testing)
                examples.append(js_example)
                js_serialized = json.dumps(js_example, ensure_ascii=False)
                f.write(js_serialized + "\n")

    print("Finished writing file {}\n".format(out_file))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-csv_to_json', type=str, default='data/wikihowAll.csv')
    parser.add_argument('-single_files', type=bool, default=False)

    args = parser.parse_args()

    if args.csv_to_json:
        tokenize_csv(args.csv_to_json)
        csv_to_json(args.csv_to_json.replace('.csv', '')+'_tokenized.csv', args.single_files)
