ALTER TABLE news ADD COLUMN temp_id SERIAL PRIMARY KEY;

alter table news add column id INTEGER;

explain analyse
UPDATE news
SET id = subquery.new_id
FROM (SELECT temp_id, row_number() over (ORDER BY datetime) as new_id from news) as subquery
WHERE news.temp_id = subquery.temp_id;

alter table news drop column temp_id;

