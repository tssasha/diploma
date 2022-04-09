from time import time
from db.db_tools import select_to_df
from src.clusterer import Clusterer
import configparser
from sklearn.metrics.cluster import v_measure_score
from sklearn import metrics

parser = configparser.ConfigParser()
test_size = 100


def func(x):
    return x.index.to_list()


for i in range(1):
    start = 10000 + i * test_size
    df = select_to_df(start, start + test_size)
    grouped = df.groupby('story_url').apply(func)

    correct_clusters = [0] * test_size
    k = 0
    for elem in grouped:
        for x in elem:
            correct_clusters[x - test_size] = k
        k += 1

    Clusterer.start_id = start
    clusterer = Clusterer()

    pred_time = time()
    for j in range(test_size):
        clusterer.clusterize_one()
        if j % 10 == 0:
            print('%s : %s' % (j, time() - pred_time))
            pred_time = time()

    predicted_clusters = [0] * test_size
    k = 0
    for cluster in clusterer.clusters:
        for x in cluster[1]:
            predicted_clusters[x - start - 1] = k
        k += 1

    print('v_measure_score:', v_measure_score(correct_clusters, predicted_clusters))
    print('homogeneity_score:', metrics.homogeneity_score(correct_clusters, predicted_clusters))
    print('completeness_score:', metrics.completeness_score(correct_clusters, predicted_clusters))
    print('adjusted_rand_score:', metrics.adjusted_rand_score(correct_clusters, predicted_clusters))
    print('adjusted_mutual_info_score:', metrics.adjusted_mutual_info_score(correct_clusters, predicted_clusters))
