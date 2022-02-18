from contextlib import contextmanager
import pandas as pd
import psycopg2
import psycopg2.extras


@contextmanager
def get_cursor():
    with psycopg2.connect(database="postgres", user='postgres', password='example', host='localhost',
                          port='5432') as conn:
        cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        try:
            yield cursor
            conn.commit()
        except:
            conn.rollback()
            raise


def select_to_df(sql):
    with psycopg2.connect(database="postgres", user='postgres', password='example', host='localhost',
                          port='5432') as conn:
        return pd.read_sql_query(sql, conn)


def select_by_id(news_id):
    with get_cursor() as cursor:
        cursor.execute('select * from news where id = %s', [news_id, ])
        return cursor.fetchone()
