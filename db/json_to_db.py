import json
from bson.json_util import dumps
import pandas as pd
from sqlalchemy import create_engine
from bson import ObjectId
from db.db_tools import get_cursor
from os.path import join


class JSONEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, ObjectId):
            return str(o)
        return super().default(o)


def parse_json():
    with open(join("db", 'new_news.json'), "r", encoding="utf8") as f:
        print("File found")
        lst = json.loads(dumps(f))
        print("Data loaded")
        ret_lst = []
        for elem in lst:
            elem = eval(elem)
            elem.pop('_id', None)
            ret_lst.append(elem)
        print("Parsing ccomplete")
        return ret_lst


def create_table():
    with get_cursor() as cursor:
        with open(join("db", 'schema.sql')) as f:
            cursor.execute(f.read())


def fill_table():
    df = pd.DataFrame(parse_json())
    engine = create_engine("postgresql+psycopg2://postgres:example@db:5432/postgres")
    print("Migration started")
    df.to_sql("news", engine, index=False, if_exists='append')
    print("Migration ended")
    with get_cursor() as cursor:
        with open(join("db", 'news_id.sql')) as f:
            cursor.execute(f.read())
    print("A new column added: finishing")


if __name__ == "__main__":
    print("Begin migrations")
    create_table()
    fill_table()
