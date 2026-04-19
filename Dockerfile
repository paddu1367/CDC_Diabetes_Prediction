FROM python:3.14-slim

WORKDIR /app

# Install dependencies first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app source and models
COPY app.py .
COPY templates/ templates/
COPY models/ models/

# Copy and run tests at build time — build fails if tests fail
COPY tests/ tests/
RUN python -m pytest tests/ -v

EXPOSE 5000

CMD ["python", "app.py"]
