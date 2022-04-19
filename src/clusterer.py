import os
import pickle
import sys
from os.path import join

import pandas as pd
import spacy
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from db.db_tools import select_to_df
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
    config.read(join('src', 'config_params.ini'))
    dict_size = config['dict'].getint('dict_size')
    start_id = max(config['DEFAULT'].getint('start_news_id') - 1, dict_size)
    dict_update_freq = config['dict'].getint('dict_update_freq')
    is_bert = config['DEFAULT'].getboolean('using_bert')

    def __init__(self):
        self.cur_id = self.start_id
        self.fitted_dicts = None
        self.clusters = None
        self.news_segment = None
        self.generate_dicts()

    def generate_dicts(self):
        directory = join('src', 'generated_dicts')
        if not os.path.exists(directory):
            os.makedirs(directory)
        start = self.cur_id + 1 - self.dict_size
        end = self.cur_id + 1
        file_path = join(f'{directory}', f'{start}_{end}.pk')
        if os.path.exists(file_path):
            with open(file_path, 'rb') as f:
                self.fitted_dicts = pickle.load(f)
                return

        logging.info("Dictionary generation in progress...")
        df = select_to_df(start, end)
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

        with open(file_path, 'wb') as f:
            pickle.dump(self.fitted_dicts, f)

    def clusterize_one(self):
        count = self.cur_id - self.start_id
        if self.cur_id != self.start_id and count % self.dict_update_freq == 0:
            self.generate_dicts()
        if count % 100 == 0:
            self.news_segment = select_to_df(self.cur_id, self.cur_id + 100).to_dict('records')
        self.cur_id += 1
        row = self.news_segment[count % 100]
        title = self.nlp(row['title'])
        body = self.nlp(row['text'])
        title_body = self.nlp(row['title'] + ' ' + row['text'])
        doc = [row['title'], lemmas(title), entities(title),
               row['text'], lemmas(body), entities(body),
               row['title'] + ' ' + row['text'], lemmas(title_body), entities(title_body)]

        mtx = [self.fitted_dicts[j].transform([doc[j]]).toarray().flatten() for j in range(9)]

        sentence_bert = None
        if self.is_bert:
            from src.sentence_bert import SentenceBert
            sentence_bert = SentenceBert(row['text'])

        if self.clusters is None:
            self.clusters = [(mtx, [self.cur_id])]
            if self.is_bert:
                self.clusters = [(mtx, [self.cur_id], sentence_bert.new_tokens)]
            return 0

        cscores = [cscore(cluster[0], mtx) for cluster in self.clusters]
        if self.is_bert:
            for i in range(len(cscores)):
                cscores[i] += sentence_bert.cosine_similarity(self.clusters[i][2])
        max_cscore = np.max(cscores)

        if self.is_bert:
            clustering_param = self.config['DEFAULT'].getfloat('bert_clustering_param')
        else:
            clustering_param = self.config['DEFAULT'].getfloat('no_bert_clustering_param')
        if max_cscore < clustering_param:
            if self.is_bert:
                self.clusters.append((mtx, [self.cur_id], sentence_bert.new_tokens))
            else:
                self.clusters.append((mtx, [self.cur_id]))
            return len(self.clusters) - 1

        cl_arg = np.argmax(cscores)
        cur_cluster = self.clusters[cl_arg]
        cl_size = len(cur_cluster[1])
        for i in range(9):
            self.clusters[cl_arg][0][i] = (cur_cluster[0][i] * cl_size + mtx[i]) / (cl_size + 1)
        self.clusters[cl_arg][1].append(self.cur_id)
        if self.is_bert:
            torch.cat((self.clusters[cl_arg][2]['input_ids'], sentence_bert.new_tokens['input_ids']), 0)
            torch.cat((self.clusters[cl_arg][2]['attention_mask'], sentence_bert.new_tokens['attention_mask']), 0)
        return cl_arg
