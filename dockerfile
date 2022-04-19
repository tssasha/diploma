FROM python:3.7


COPY requirements.txt /app/
WORKDIR "/app"
RUN pip install -r /app/requirements.txt
RUN python -m spacy download ru_core_news_sm
COPY ./db /app/db
COPY ./db_requirements.txt /app/
RUN pip install -r /app/db_requirements.txt
COPY ./src /app/src
ENTRYPOINT []
