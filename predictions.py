import numpy as np
import pandas as pd
import boto3
from tensorflow.keras.models import load_model

# Read AWS credentials from environment variables (GitHub Actions will set these)
aws_access_key_id = os.environ.get('AWS_ACCESS_KEY_ID')
aws_secret_access_key = os.environ.get('AWS_SECRET_ACCESS_KEY')
aws_region = os.environ.get('AWS_DEFAULT_REGION', 'your-default-region')  # Optional

# Initialize S3 client with credentials
s3 = boto3.client('s3',
                  aws_access_key_id=aws_access_key_id,
                  aws_secret_access_key=aws_secret_access_key,
                  region_name=aws_region)

# Download the trained model from S3 (if not already done)
s3.download_file('miscbucket-saraththarayil', 'trained_model.h5', 'trained_model.h5')

# Download the test data from S3
s3.download_file('miscbucket-saraththarayil', 'test_data.npz', 'test_data.npz')

# Load the pre-trained Keras model
model = load_model('trained_model.h5')

# Load the test data from the downloaded .npz file
data = np.load('test_data.npz')

# Extract the required arrays
X_test = data['X_test']

print("Original shape of X_test:", X_test.shape)

# Reshape if necessary
X_test_reshaped = X_test.reshape(X_test.shape[0], 26, 24)

print("Reshaped X_test shape:", X_test_reshaped.shape)

# Perform inference
predictions = model.predict(X_test_reshaped)

# Save predictions to a CSV file
predictions_df = pd.DataFrame(predictions)
predictions_df.to_csv('predictions.csv', index=False)

# Optionally, upload predictions to S3
predictions_df.to_csv('/tmp/predictions.csv', index=False)
s3.upload_file('/tmp/predictions.csv', 'miscbucket-saraththarayil', 'predictions.csv')

print("Predictions have been saved and uploaded to 'predictions.csv'")
