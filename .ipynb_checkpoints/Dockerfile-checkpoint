# Use the official lightweight Python image from
# https://hub.docker.com/_/python
FROM python:3.8-slim

# Copy all the files needed for the app to work
COPY inference.py .
COPY best_estimator/ ./best_estimator

# Install all the necessary libraries
RUN pip install -r ./best_estimator/requirements.txt

# Run the API!
CMD python inference.py
