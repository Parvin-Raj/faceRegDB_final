# Use official Python base image
FROM python:3.9-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libaio1 \
    unzip \
    wget \
 && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy your local Oracle Instant Client ZIP instead of downloading
COPY instantclient-basic-linux.x64-23.8.0.25.04.zip /opt/oracle/

# Setup Oracle Instant Client
WORKDIR /opt/oracle
RUN unzip instantclient-basic-linux.x64-23.8.0.25.04.zip && \
    rm instantclient-basic-linux.x64-23.8.0.25.04.zip

# Set Oracle Instant Client environment variables for version 23.8
ENV ORACLE_HOME=/opt/oracle/instantclient_23_8
ENV LD_LIBRARY_PATH=$ORACLE_HOME
ENV PATH=$ORACLE_HOME:$PATH

# Return to app directory
WORKDIR /app

# Copy your application code
COPY . .

# Expose Flask port
EXPOSE 5000

# Run the Flask app
CMD ["python", "app.py"]
