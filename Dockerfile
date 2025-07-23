FROM python:3.11-slim

# Install llama-cpp and system-level dependencies
RUN apt-get update && apt-get install -y \
    cmake \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set workdir inside container
WORKDIR /app

# Copy all project files into the image
COPY . .

# Install Python dependencies
RUN pip install --upgrade pip && pip install -r requirements.txt

# Expose port for FastAPI
EXPOSE 8000

# Run ingestion, then launch FastAPI
CMD ["bash", "-c", "python app/ingestion.py && python main.py"]
