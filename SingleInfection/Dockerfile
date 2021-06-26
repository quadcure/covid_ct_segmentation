FROM python:3.7.10
RUN mkdir /ctlunginfection
ADD ./* /ctlunginfection/
WORKDIR /ctlunginfection

RUN pip install -r requirements.txt

EXPOSE 5000

CMD ["python", "LungInfection.py"]
