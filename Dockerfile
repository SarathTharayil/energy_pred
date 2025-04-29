# Use a lightweight base image
FROM python:3.8-slim

# Set the working directory
WORKDIR /app

# Copy only your Python script
COPY predictions.py /app/predict_model.py

# Install necessary Python packages
RUN pip install --no-cache-dir numpy pandas tensorflow boto3

# Command to run the script
CMD ["python", "predict_model.py"]
