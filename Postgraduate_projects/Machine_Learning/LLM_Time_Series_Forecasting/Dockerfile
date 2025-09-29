# Use an official Python image
FROM python:3.10-slim

# Set working directory inside the container
WORKDIR /app

# Copy requirements and install them
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all source code and scripts into the container
COPY . .

# Default command (can be overridden)
CMD ["python", "src/q2_1_visualise_data.py"]