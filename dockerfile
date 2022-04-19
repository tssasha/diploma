FROM python:3.7


COPY requirements.txt /app/
WORKDIR "/app"
RUN pip install -r /app/requirements.txt
RUN python -m spacy download ru_core_news_sm
COPY ./db /app/db
COPY ./db_requirements.txt /app/
RUN pip install -r /app/db_requirements.txt
RUN \
  apt-get update && \
  apt-get install -y sudo curl git && \
  curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash && \
  sudo apt-get install git-lfs
RUN git clone https://huggingface.co/DeepPavlov/rubert-base-cased
COPY ./src /app/src
ENTRYPOINT []
