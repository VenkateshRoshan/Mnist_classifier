# Base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy necessary files
COPY requirements.txt .
COPY . .

# Install dependencies
RUN pip install -r requirements.txt

# Expose port for Gradio
EXPOSE 7860

# Run the Gradio app
CMD ["python", "app.py"]
