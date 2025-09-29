FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and all modules (including module_3 with files)
COPY requirements.txt .
COPY module_1/ ./module_1/
COPY module_2/ ./module_2/
COPY module_3/ ./module_3/

# Install Python packages
RUN pip install --upgrade pip && pip install -r requirements.txt && pip install jupyter

# Expose Jupyter port
EXPOSE 8888

# Start Jupyter Notebook
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--allow-root", "--NotebookApp.token=''", "--NotebookApp.notebook_dir=/app"]
