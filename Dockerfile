FROM python:3.10-slim

# Install essential packages for audio processing
RUN apt-get update && apt-get install -y \
    build-essential \
    libsndfile1 \
    libespeak-dev \
    espeak \
    ffmpeg \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Rust compiler (needed for sudachipy)
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Set working directory
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install additional dependencies for audio processing
RUN pip install --no-cache-dir pydub

# Copy the rest of the application
COPY . .

# Create a directory for session files
RUN mkdir -p /app/sessions

# Command to run the bot
CMD ["python", "bot.py"] 