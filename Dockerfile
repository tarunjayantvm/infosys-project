# Use an official Python 3.10 image as the base
FROM python:3.10-slim

# Set the folder where your app will live inside the container
WORKDIR /app

# Copy the list of Python packages needed
COPY requirements.txt .

# Install the Python packages. We install torch separately for CPU for compatibility.
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Copy all of your project files into the container
COPY . .

# Tell Hugging Face that your app will run on port 7860
EXPOSE 7860

# The command to start your web server when the container runs
# This runs your Flask app 'app' from the file 'app.py'
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "--workers", "1", "app:app"]