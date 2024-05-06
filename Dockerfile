# Use an official Python runtime as a parent image
FROM python:3.9
LABEL authors="ouail laamiri"

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "main.py", "main:app"]