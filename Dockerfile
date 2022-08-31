# For more information, please refer to https://aka.ms/vscode-docker-python
FROM python:3.9-slim AS base

EXPOSE 50051

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

RUN  apt update && apt install libcurl4 -y && rm -rf /var/lib/apt/lists/*

# Install pip requirements
COPY requirements.txt .
RUN python -m pip install -r requirements.txt --no-cache-dir
RUN python -m pip install fastapi==0.75.2 --no-cache-dir

FROM base as app


WORKDIR /app

# Creates a non-root user with an explicit UID and adds permission to access the /app folder
# For more info, please refer to https://aka.ms/vscode-docker-python-configure-containers
RUN adduser -u 5678 --disabled-password --gecos "" appuser

COPY --chown=appuser . /app

USER appuser

ENV PYTHONPATH=/app

# During debugging, this entry point will be overridden. For more information, please refer to https://aka.ms/vscode-docker-python-debug
CMD ["python", "lego_sorter_server/__main__.py"]
