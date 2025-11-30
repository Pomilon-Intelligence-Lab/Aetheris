FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Create a user first to handle permissions correctly from the start
RUN useradd -m -u 1000 user

# Switch to user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Set up application directory with correct permissions
WORKDIR $HOME/app

# Copy requirements and install
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

# Copy application code
COPY --chown=user . .

# Expose port
EXPOSE 7860

# Command to run the application
CMD ["python3", "-m", "aetheris.cli.main", "serve", "--host", "0.0.0.0", "--port", "7860"]
