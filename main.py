from flask import Flask, render_template
from src.clusterer import Clusterer
from db.db_tools import select_by_id

app = Flask(__name__)


@app.route('/')
def index():
    return render_template("main.html", news_id=None, news_title=None, result=None)


@app.route('/clusterize_one/', methods=["POST"])
def clusterize_one():
    ans = clusterer.clusterize_one()
    news_title = select_by_id(clusterer.cur_id)['title']
    return render_template("main.html", news_id=clusterer.cur_id, news_title=news_title, result=str(ans))


if __name__ == '__main__':
    clusterer = Clusterer()
    app.run(host='127.0.0.1', port=8000)
