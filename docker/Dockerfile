# Use a CUDA-enabled Python base image (small, optimized)
FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    ffmpeg libsm6 libxext6 && \
    rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy only the necessary files first (for better caching)
COPY requirements.txt .

# Install Python dependencies (with no cache)
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project
COPY . .

# Expose port if needed (optional)
# EXPOSE 8000

# Default command (can be overridden at runtime)
ENTRYPOINT ["python3"]
CMD ["detect_missing_new_objects.py"]
