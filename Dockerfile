FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

# 先升級 pip 再安裝套件
RUN pip install --upgrade pip --no-cache-dir && \
    pip install --no-cache-dir -r requirements.txt


COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]