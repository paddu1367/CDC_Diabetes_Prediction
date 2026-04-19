FROM python:3.14-slim

WORKDIR /app

# Install dependencies first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app source
COPY app.py .
COPY templates/ templates/
COPY models/ models/

EXPOSE 5000

CMD ["python", "app.py"]
