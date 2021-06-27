FROM python:3.7.10

RUN mkdir /SingleInfection
ADD ./* /SingleInfection/
WORKDIR /SingleInfection

RUN pip install -r requirements.txt

EXPOSE 8001

CMD ["uvicorn", "SingleInfection:app", "--host", "0.0.0.0", "--port", "8001"]
