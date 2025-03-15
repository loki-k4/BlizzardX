# Use a Python base image
FROM python:3.11

# Set the working directory in the container
WORKDIR /workspace

# Install dependencies and other necessary tools
RUN apt-get update && \
    apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set up the virtual environment
RUN python -m venv /workspace/venv
ENV PATH="/workspace/venv/bin:$PATH"

# Copy the requirements file and install dependencies
COPY requirements.txt /workspace/requirements.txt
RUN pip install --cache-dir=/tmp/.cache -r /workspace/requirements.txt

# Expose port for any possible web servers (adjust as needed)
EXPOSE 8000

# Set the default command to open a bash shell
CMD ["bash"]
