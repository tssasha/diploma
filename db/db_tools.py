from contextlib import contextmanager

import psycopg2


@contextmanager
def get_session():
    with psycopg2.connect(database="postgres", user='postgres', password='example', host='localhost',
                          port='5432') as conn:
        cursor = conn.cursor()
        try:
            yield cursor
            conn.commit()
        except:
            conn.rollback()
            raise
