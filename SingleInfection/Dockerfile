FROM python:3.6.10
RUN mkdir /app
ADD ./* /app/
WORKDIR /app

RUN pip install -r requirements.txt

EXPOSE 5000

CMD ["python", "LungInfection.py"]
