# Use a specific version of the python image
FROM python:3.12-bookworm

# Set environment variables
ENV PYTHONUNBUFFERED True \
    APP_HOME /app

# Set the working directory
WORKDIR $APP_HOME

# Install system dependencies for your Python project
# And clean up the cache to keep the image slim
RUN apt-get update && apt-get install -y \
    && rm -rf /var/lib/apt/lists/*


# Copy only the requirements file first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Expose the port the app runs on
EXPOSE 8000

# Start the application
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]

# CMD ["gunicorn", "app.main:app", "--workers", "4", "--worker-class", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000"]
# gunicorn app.main:app --workers 4 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000