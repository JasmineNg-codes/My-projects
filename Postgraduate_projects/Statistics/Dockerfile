# Use a Python 3.9.6 base image
FROM python:3.9.6-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements.txt file to the container
COPY requirements.txt .

# Update pip to the latest version
RUN python -m pip install --upgrade pip

# Install all the dependencies listed in requirements.txt
RUN pip install -r requirements.txt

# Install Jupyter Notebook for running the notebook
RUN pip install notebook

# Copy the notebook and any necessary files into the container
COPY S1.ipynb /app/

# Expose the Jupyter Notebook default port
EXPOSE 8888

# Set the default command to run Jupyter Notebook
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
