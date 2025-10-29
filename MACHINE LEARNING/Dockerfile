# Use the base image prepared for the course
FROM agrigorev/zoomcamp-model:2025

# Set working directory inside the container
WORKDIR /app

# Copy your project files into the container
COPY . .

# Install uv and dependencies
RUN pip install uv && uv sync --frozen

# Expose FastAPI port
EXPOSE 8000

# Run FastAPI with Uvicorn
CMD ["uv", "run", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
