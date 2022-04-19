import os
from os.path import join

from flask import Flask, render_template
from src.clusterer import Clusterer
from db.db_tools import select_by_id

template_dir = os.path.abspath(join('src', 'templates'))
app = Flask(__name__, template_folder=template_dir)


@app.route('/')
def index():
    return render_template("main.html", news_id=None, news_title=None, result=None)


@app.route('/clusterize_one/', methods=["POST"])
def clusterize_one():
    ans = clusterer.clusterize_one()
    news_title = select_by_id(clusterer.cur_id)['title']
    cluster_news = []
    for id in clusterer.clusters[ans][1]:
        cluster_news.append(select_by_id(id)['title'])
    return render_template("main.html", news_id=clusterer.cur_id, news_title=news_title, result=str(ans), cluster_news=cluster_news)


if __name__ == '__main__':
    clusterer = Clusterer()
    app.run(host='0.0.0.0', port=8000)
