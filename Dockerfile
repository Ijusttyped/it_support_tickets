FROM python:latest
ADD . /code
WORKDIR /code
RUN pip install -r requirements.txt
CMD python src/api/main.py
EXPOSE 5000