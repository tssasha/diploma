from time import time
from db.db_tools import select_to_df
from src.clusterer import Clusterer
from sklearn.metrics.cluster import v_measure_score
from sklearn import metrics

test_size = 100
iterations = 1


def func(x):
    return x.index.to_list()


avg_v_measure_score = 0
avg_homogeneity_score = 0
avg_completeness_score = 0
avg_adjusted_rand_score = 0
avg_adjusted_mutual_info_score = 0

all_time = 0

for i in range(iterations):
    start = i * 100000 + 10000
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
        if j % (test_size / 10) == 0:
            all_time += time() - pred_time
            print('%s : %s' % (j, time() - pred_time))
            pred_time = time()

    predicted_clusters = [0] * test_size
    k = 0
    for cluster in clusterer.clusters:
        for x in cluster[1]:
            predicted_clusters[x - start - 1] = k
        k += 1

    vmeasure_score = v_measure_score(correct_clusters, predicted_clusters)
    homogeneity_score = metrics.homogeneity_score(correct_clusters, predicted_clusters)
    completeness_score = metrics.completeness_score(correct_clusters, predicted_clusters)
    adjusted_rand_score = metrics.adjusted_rand_score(correct_clusters, predicted_clusters)
    adjusted_mutual_info_score = metrics.adjusted_mutual_info_score(correct_clusters, predicted_clusters)

    print('v_measure_score: ', vmeasure_score)
    print('homogeneity_score: ', homogeneity_score)
    print('completeness_score: ', completeness_score)
    print('adjusted_rand_score: ', adjusted_rand_score)
    print('adjusted_mutual_info_score: ', adjusted_mutual_info_score)

    avg_v_measure_score += vmeasure_score
    avg_homogeneity_score += homogeneity_score
    avg_completeness_score += completeness_score
    avg_adjusted_rand_score += adjusted_rand_score
    avg_adjusted_mutual_info_score += adjusted_mutual_info_score

print('avg_v_measure_score:', avg_v_measure_score / iterations)
print('avg_homogeneity_score:', avg_homogeneity_score / iterations)
print('avg_completeness_score:', avg_completeness_score / iterations)
print('avg_adjusted_rand_score:', avg_adjusted_rand_score / iterations)
print('avg_adjusted_mutual_info_score:', avg_adjusted_mutual_info_score / iterations)
print('time for clusterize_one : ', all_time / test_size / iterations)
