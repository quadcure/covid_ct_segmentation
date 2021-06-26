FROM python:3.7.10

RUN mkdir /ctlungmulticlass
ADD ./* /ctlungmulticlass/
WORKDIR /ctlungmulticlass

RUN pip install -r requirements.txt

EXPOSE 8000

CMD ["uvicorn", "LungInfection:app", "--host", "0.0.0.0", "--port", "8000"]

