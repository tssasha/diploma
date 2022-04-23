import logging
import sys

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from db.db_tools import select_to_df
from src.clusterer import Clusterer

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
train_start = 510000
train_corpus_size = 100
reserve_size = 1


def func(x):
    return x.index.to_list()


def generate_corpus():
    df = select_to_df(train_start + 1, train_start + 1 + reserve_size + train_corpus_size)
    clusterer = Clusterer(10000, train_start, True)

    available_clusters = []
    y = []
    x = []

    for i in range(reserve_size):
        story_url = df.iloc[i]['story_url']
        if story_url in available_clusters:
            clusterer.clusterize_one_for_ccm(available_clusters.index(story_url))
        else:
            clusterer.clusterize_one_for_ccm(len(available_clusters))
            available_clusters.append(story_url)

    for i in range(reserve_size, train_corpus_size + reserve_size):
        story_url = df.iloc[i]['story_url']
        if story_url in available_clusters:
            y.append(0)
            x.append(clusterer.clusterize_one_for_ccm(available_clusters.index(story_url)))
        else:
            y.append(1)
            x.append(clusterer.clusterize_one_for_ccm(len(available_clusters)))
            available_clusters.append(story_url)

    return train_test_split(x, y, test_size=0.2, random_state=0)


def fit_model(x, y):
    return LogisticRegression(solver='lbfgs', random_state=0).fit(x, y)


def create_model():
    logging.info('Cluster creation model generation in progress...')
    x_tr, _, y_tr, _ = generate_corpus()
    cc_model = fit_model(x_tr, y_tr)
    logging.info('Cluster creation model generated')
    return cc_model


if __name__ == '__main__':
    x_train, x_test, y_train, y_test = generate_corpus()
    model = fit_model(x_train, y_train)
    y_pred = model.predict(x_test)
    print(model.score(x_train, y_train))
    print(model.score(x_test, y_test))
    print(confusion_matrix(y_test, y_pred))
