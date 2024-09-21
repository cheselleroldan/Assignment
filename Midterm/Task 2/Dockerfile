# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Install system dependencies including a C++ compiler
RUN apt-get update && apt-get install -y \
    libmupdf-dev \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables to ensure the Python output is not buffered
ENV PYTHONUNBUFFERED=1

# Set the correct HOME and PATH variables
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Create a new non-root user and set permissions
RUN useradd -m user

# Set the working directory in the container
WORKDIR $HOME/app

# Copy the current directory contents into the container's working directory
COPY . .

# Upgrade pip and install any needed packages specified in requirements.txt
RUN pip install --upgrade pip
# Upgrade pip and install packages using the legacy resolver
RUN pip install --default-timeout=1000 --no-cache-dir --use-deprecated=legacy-resolver -r requirements.txt


# Expose the port Chainlit runs on (by default Chainlit uses port 8000)
EXPOSE 8000

# Run the app
CMD ["chainlit", "run", "app.py", "--host", "0.0.0.0"]
