# For more information, please refer to https://aka.ms/vscode-docker-python
FROM python:3.9-slim AS base

EXPOSE 50051

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

RUN  apt update && apt install libcurl4 wget unzip -y && rm -rf /var/lib/apt/lists/*

# Install pip requirements
COPY requirements.txt .
RUN python -m pip install -r requirements.txt --no-cache-dir
# RUN python -m pip install fastapi==0.75.2 --no-cache-dir

FROM base as app


WORKDIR /app

# Creates a non-root user with an explicit UID and adds permission to access the /app folder
# For more info, please refer to https://aka.ms/vscode-docker-python-configure-containers
RUN adduser -u 5678 --disabled-password --gecos "" appuser

RUN wget https://github.com/LegoSorter/LegoSorterServer/releases/download/1.2.0/detection_models.zip
RUN wget https://github.com/LegoSorter/LegoSorterServer/releases/download/1.2.0/classification_models.zip
RUN wget https://github.com/LegoSorter/LegoSorterServer/releases/download/1.3.0-alpha/yolov5_n.zip
RUN wget https://github.com/LegoSorter/LegoSorterServer/releases/download/1.3.0-alpha/tinyvit.zip

RUN mkdir -p ./lego_sorter_server/analysis/detection/models/yolo_model
RUN mkdir -p ./lego_sorter_server/analysis/classification/models/tiny_vit_model

RUN unzip detection_models.zip -d ./lego_sorter_server/analysis/detection/models
RUN unzip classification_models.zip -d ./lego_sorter_server/analysis/classification/models

RUN unzip yolov5_n.zip -d ./lego_sorter_server/analysis/detection/models/yolo_model/
RUN unzip tinyvit.zip -d ./lego_sorter_server/analysis/classification/models/tiny_vit_model

RUN rm detection_models.zip
RUN rm classification_models.zip
RUN rm yolov5_n.zip
RUN rm tinyvit.zip

RUN chown -R appuser ./lego_sorter_server/analysis/detection/models/
RUN chown -R appuser ./lego_sorter_server/analysis/classification/models/

COPY --chown=appuser . /app

USER appuser

ENV PYTHONPATH=/app

# During debugging, this entry point will be overridden. For more information, please refer to https://aka.ms/vscode-docker-python-debug
CMD ["python", "lego_sorter_server/__main__.py"]
