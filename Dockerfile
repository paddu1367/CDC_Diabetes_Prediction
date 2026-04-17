FROM python:3.14-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "-c", "import pandas as pd; import matplotlib.pyplot as plt; import seaborn as sns; import numpy as np; print('All libraries loaded successfully')"]