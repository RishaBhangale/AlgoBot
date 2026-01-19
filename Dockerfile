# Use Python 3.9 slim image
FROM python:3.9-slim

# Install Chromium browser and ChromeDriver
RUN apt-get update && apt-get install -y \
    wget \
    gnupg \
    unzip \
    chromium \
    chromium-driver \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables for Chrome
ENV CHROME_BIN=/usr/bin/chromium
ENV CHROMEDRIVER_PATH=/usr/bin/chromedriver

# Set working directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 10000

# Run with uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "10000"]
