import logging
import sys
from time import time
from db.db_tools import select_to_df
from src.cluster_creation_model import create_model
from src.clusterer import Clusterer
from sklearn.metrics.cluster import v_measure_score
from sklearn import metrics

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
test_size = 1000
iterations = 10


def func(x):
    return x.index.to_list()


if __name__ == '__main__':
    avg_v_measure_score = 0
    avg_homogeneity_score = 0
    avg_completeness_score = 0
    avg_adjusted_rand_score = 0
    avg_adjusted_mutual_info_score = 0

    all_time = 0
    cc_model = create_model()

    for i in range(iterations):
        dict_size = 10000
        start = i * 100000 + dict_size + 1
        df = select_to_df(start, start + test_size)
        grouped = df.groupby('story_url').apply(func)

        correct_clusters = [0] * test_size
        k = 0
        for elem in grouped:
            for x in elem:
                correct_clusters[x] = k
            k += 1

        clusterer = Clusterer(dict_size=dict_size, start_id=start - 1, cc_model=cc_model)

        pred_time = time()
        for j in range(test_size):
            clusterer.clusterize_one()
            if j % (test_size / 10) == 0:
                all_time += time() - pred_time
                print('%s : %s' % (j, time() - pred_time))
                pred_time = time()

        predicted_clusters = [0] * test_size
        k = 0
        for cluster in clusterer.clusters:
            for x in cluster[1]:
                predicted_clusters[x - start] = k
            k += 1

        vmeasure_score = v_measure_score(correct_clusters, predicted_clusters)
        homogeneity_score = metrics.homogeneity_score(correct_clusters, predicted_clusters)
        completeness_score = metrics.completeness_score(correct_clusters, predicted_clusters)
        adjusted_rand_score = metrics.adjusted_rand_score(correct_clusters, predicted_clusters)
        adjusted_mutual_info_score = metrics.adjusted_mutual_info_score(correct_clusters, predicted_clusters)

        logging.info('v_measure_score: %s' % vmeasure_score)
        logging.info('homogeneity_score: %s' % homogeneity_score)
        logging.info('completeness_score: %s' % completeness_score)
        logging.info('adjusted_rand_score: %s' % adjusted_rand_score)
        logging.info('adjusted_mutual_info_score: %s' % adjusted_mutual_info_score)

        avg_v_measure_score += vmeasure_score
        avg_homogeneity_score += homogeneity_score
        avg_completeness_score += completeness_score
        avg_adjusted_rand_score += adjusted_rand_score
        avg_adjusted_mutual_info_score += adjusted_mutual_info_score

    logging.info('avg_v_measure_score: %s' % (avg_v_measure_score / iterations))
    logging.info('avg_homogeneity_score: %s' % (avg_homogeneity_score / iterations))
    logging.info('avg_completeness_score: %s' % (avg_completeness_score / iterations))
    logging.info('avg_adjusted_rand_score: %s' % (avg_adjusted_rand_score / iterations))
    logging.info('avg_adjusted_mutual_info_score: %s' % (avg_adjusted_mutual_info_score / iterations))
    logging.info('time for clusterize_one: %s' % (all_time / test_size / iterations))
