FROM python:3.7.10

RUN mkdir /MultiInfection
ADD ./* /MultiInfection/
WORKDIR /MultiInfection

RUN pip install -r requirements.txt

EXPOSE 8002

CMD ["uvicorn", "MultiInfection:app", "--host", "0.0.0.0", "--port", "8002"]

