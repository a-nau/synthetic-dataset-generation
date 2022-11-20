FROM python:3.9-slim-bullseye


RUN apt-get -y update

# see https://serverfault.com/a/992421
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ="America/New_York"

RUN apt-get install -y libopencv-dev python3-opencv git cmake

RUN mkdir -p /app
WORKDIR /app

RUN pip install --upgrade pip
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

ENV PYTHONPATH "${PYTHONPATH}:/app"

CMD ["python", "src/tools/generate_synthetic_data.py"]
