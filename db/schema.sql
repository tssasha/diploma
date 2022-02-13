DROP TABLE IF EXISTS news;

CREATE TABLE news (
    title TEXT NOT NULL,
    url TEXT,
    story_url TEXT NOT NULL,
    text TEXT NOT NULL,
    agency TEXT,
    category TEXT,
    platform TEXT,
    type TEXT,
    datetime INTEGER NOT NULL
);