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


def sim_metrics(cluster_mtx, doc_mtx):
    return [1 - spatial.distance.cosine(cluster_mtx[i], doc_mtx[i]) for i in range(9)]


logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)


class Clusterer:
    nlp = spacy.load("ru_core_news_sm")
    config = configparser.ConfigParser()
    config.read(join('src', 'config_params.ini'))
    is_bert = config['DEFAULT'].getboolean('using_bert')
    dict_size = config['dict'].getint('dict_size')
    start_id = max(config['DEFAULT'].getint('start_news_id') - 1, dict_size)

    def __init__(self, dict_size=dict_size, start_id=start_id, for_ccm_creation=False):
        self.dict_size = dict_size
        self.start_id = start_id
        self.for_ccm_creation = for_ccm_creation
        self.cur_id = self.start_id
        self.fitted_dicts = None
        self.clusters = None
        self.news_segment = None
        self.generate_dicts()
        if not for_ccm_creation:
            from src.cluster_creation_model import create_model
            self.cc_model = create_model()

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

    def clusterize_one_for_ccm(self, right_cluster_num):
        mtx, sentence_bert = self.generate_document_vectors()

        if self.clusters is None:
            self.clusters = [(mtx, [self.cur_id])]
            if self.is_bert:
                self.clusters = [(mtx, [self.cur_id], sentence_bert.new_tokens)]
            return None

        if right_cluster_num >= len(self.clusters):
            metrix = [sim_metrics(cluster[0], mtx) for cluster in self.clusters]
            cscores = [sum(m) for m in metrix]
            if self.is_bert:
                for i in range(len(cscores)):
                    bert_sim = sentence_bert.cosine_similarity(self.clusters[i][2])
                    metrix[i].append(bert_sim)
                    cscores[i] += bert_sim
                self.clusters.append((mtx, [self.cur_id], sentence_bert.new_tokens))
            else:
                self.clusters.append((mtx, [self.cur_id]))

            cl_arg = np.argmax(cscores)
            return metrix[cl_arg]
        else:
            metrix = sim_metrics(self.clusters[right_cluster_num][0], mtx)
            if self.is_bert:
                metrix.append(sentence_bert.cosine_similarity(self.clusters[right_cluster_num][2]))
            self.add_document_to_cluster(right_cluster_num, mtx, sentence_bert)
            return metrix

    def clusterize_one(self):
        mtx, sentence_bert = self.generate_document_vectors()

        if self.clusters is None:
            self.clusters = [(mtx, [self.cur_id])]
            if self.is_bert:
                self.clusters = [(mtx, [self.cur_id], sentence_bert.new_tokens)]
            return 0

        metrix = [sim_metrics(cluster[0], mtx) for cluster in self.clusters]
        cscores = [sum(m) for m in metrix]
        if self.is_bert:
            for i in range(len(cscores)):
                cscores[i] += sentence_bert.cosine_similarity(self.clusters[i][2])

        cl_arg = np.argmax(cscores)
        max_metrix = metrix[cl_arg]
        ans = self.cc_model.predict([max_metrix])[0]
        if ans == 1:
            if self.is_bert:
                self.clusters.append((mtx, [self.cur_id], sentence_bert.new_tokens))
            else:
                self.clusters.append((mtx, [self.cur_id]))
            return len(self.clusters) - 1

        self.add_document_to_cluster(cl_arg, mtx, sentence_bert)
        return cl_arg

    def generate_document_vectors(self):
        count = self.cur_id - self.start_id
        self.cur_id += 1
        if count % 100 == 0:
            self.news_segment = select_to_df(self.cur_id, self.cur_id + 100).to_dict('records')
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
        return mtx, sentence_bert

    def add_document_to_cluster(self, cl_arg, mtx, sentence_bert):
        cur_cluster = self.clusters[cl_arg]
        cl_size = len(cur_cluster[1])
        for i in range(9):
            self.clusters[cl_arg][0][i] = (cur_cluster[0][i] * cl_size + mtx[i]) / (cl_size + 1)
        self.clusters[cl_arg][1].append(self.cur_id)
        if self.is_bert:
            torch.cat((self.clusters[cl_arg][2]['input_ids'], sentence_bert.new_tokens['input_ids']), 0)
            torch.cat((self.clusters[cl_arg][2]['attention_mask'], sentence_bert.new_tokens['attention_mask']), 0)
