FROM python:3.10-slim

WORKDIR /app

# Install system dependencies for audio
RUN apt-get update && apt-get install -y \
    portaudio19-dev \
    alsa-utils \
    pulseaudio-utils \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "1", "app:app"]
