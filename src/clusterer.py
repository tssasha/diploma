import sys
import pandas as pd
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from db.db_tools import select_to_df, select_by_id
import numpy as np
from scipy import spatial
import configparser
import logging


def lemmas(lst):
    return " ".join([token.lemma_ for token in lst])


def entities(lst):
    return " ".join([ent.lemma_ for ent in lst.ents])


def cscore(cluster_mtx, doc_mtx):
    return sum(1 - spatial.distance.cosine(cluster_mtx[i], doc_mtx[i]) for i in range(9))


logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)


class Clusterer:
    nlp = spacy.load("ru_core_news_sm")
    config = configparser.ConfigParser()
    config.read('config_params.ini')
    dict_size = config['dict'].getint('dict_size')
    start_id = max(config['DEFAULT'].getint('start_news_id') - 1, dict_size)
    dict_update_freq = config['dict'].getint('dict_update_freq')

    def __init__(self):
        self.cur_id = self.start_id
        self.fitted_dicts = None
        self.clusters = None
        self.generate_dicts()

    def generate_dicts(self):
        logging.info("Dictionary generation in progress...")
        df = select_to_df(self.cur_id + 1 - self.dict_size, self.cur_id + 1)
        tokens_df = pd.DataFrame()
        tokens_df['title_nlp'] = df['title'].map(self.nlp)
        tokens_df['body_nlp'] = df['text'].map(self.nlp)

        tokens_df['t_tokens'] = df['title']
        tokens_df['t_lemmas'] = tokens_df['title_nlp'].map(lemmas)
        tokens_df['t_entities'] = tokens_df['title_nlp'].map(entities)
        tokens_df['b_tokens'] = df['text']
        tokens_df['b_lemmas'] = tokens_df['body_nlp'].map(lemmas)
        tokens_df['b_entities'] = tokens_df['body_nlp'].map(entities)

        self.fitted_dicts = [TfidfVectorizer().fit(tokens_df['t_tokens']),
                             TfidfVectorizer().fit(tokens_df['t_lemmas']),
                             TfidfVectorizer().fit(tokens_df['t_entities']),
                             TfidfVectorizer().fit(tokens_df['b_tokens']),
                             TfidfVectorizer().fit(tokens_df['b_lemmas']),
                             TfidfVectorizer().fit(tokens_df['b_entities']),
                             TfidfVectorizer().fit(tokens_df['t_tokens'] + ' ' + tokens_df['b_tokens']),
                             TfidfVectorizer().fit(tokens_df['t_lemmas'] + ' ' + tokens_df['b_lemmas']),
                             TfidfVectorizer().fit(tokens_df['t_entities'] + ' ' + tokens_df['b_entities'])]
        logging.info("Dictionary generated")

    def clusterize_one(self):
        if self.cur_id != self.start_id and (self.cur_id - self.start_id) % self.dict_update_freq == 0:
            self.generate_dicts()
        self.cur_id += 1
        row = select_by_id(self.cur_id)
        title = self.nlp(row['title'])
        body = self.nlp(row['text'])
        title_body = self.nlp(row['title'] + ' ' + row['text'])
        doc = [row['title'], lemmas(title), entities(title),
               row['text'], lemmas(body), entities(body),
               row['title'] + ' ' + row['text'], lemmas(title_body), entities(title_body)]

        mtx = [self.fitted_dicts[j].transform([doc[j]]) for j in range(9)]  #

        if self.clusters is None:
            self.clusters = [(mtx, [self.cur_id])]
            return 0

        cscores = [cscore(cluster[0], mtx) for cluster in self.clusters]
        max_cscore = np.max(cscores)

        if max_cscore < 1.5:
            self.clusters.append((mtx, [self.cur_id]))
            return len(self.clusters) - 1

        cl_arg = np.argmax(cscores)
        cur_cluster = self.clusters[cl_arg]
        cl_size = len(cur_cluster[1])
        for i in range(9):
            self.clusters[cl_arg][0][i] = (cur_cluster[0][i] * cl_size + mtx[i]) / (cl_size + 1)
        self.clusters[cl_arg][1].append(self.cur_id)
        return cl_arg
