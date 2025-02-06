FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Install Git
RUN apt-get update && apt-get install -y git

# Copy the project files into the container
COPY . .

# Remove pywin32 (Only needed for Windows, not Linux)
RUN sed -i '/pywin32/d' requirements.txt

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the application port (if needed)
EXPOSE 5000

# Command to run the app
CMD ["python", "app.py"]
