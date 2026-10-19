# Step 6: Container image for the prediction API
# Small official Python image (slim = fewer unneeded packages)
FROM python:3.10-slim

# All following commands run inside /app in the container
WORKDIR /app

# Copy only the lightweight serving requirements first. Docker caches this
# layer, so rebuilds are fast unless requirements change.
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# Copy the API code and the trained model into the image
COPY api/ api/
COPY models/model.joblib models/model.joblib

# Document that the app listens on port 8000
EXPOSE 8000

# Start the web server. 0.0.0.0 makes it reachable from outside the container.
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
