import json
from bson.json_util import dumps
import pandas as pd
from sqlalchemy import create_engine
from bson import ObjectId
from db.db_tools import get_session


class JSONEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, ObjectId):
            return str(o)
        return super().default(o)


def parse_json():
    with open('new_news.json', "r", encoding="utf8") as f:
        lst = json.loads(dumps(f))
        ret_lst = []
        for elem in lst:
            elem = eval(elem)
            elem.pop('_id', None)
            ret_lst.append(elem)
        return ret_lst


def create_table():
    with get_session() as cursor:
        with open('schema.sql') as f:
            cursor.execute(f.read())
        with open('news_id.sql') as f:
            cursor.execute(f.read())


def fill_table():
    df = pd.DataFrame(parse_json())
    engine = create_engine("postgresql+psycopg2://postgres:example@localhost:5432/postgres")
    df.to_sql("news", engine, index=False, if_exists='append')


create_table()
fill_table()
