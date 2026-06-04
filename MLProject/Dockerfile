FROM python:3.12-slim

WORKDIR /app

# Copy kebutuhan file ke dalam Docker
COPY my_model/ /app/model/
COPY crop_preprocessing.csv /app/

# Install library utama untuk serve model
RUN pip install --no-cache-dir mlflow scikit-learn pandas numpy

# Expose port untuk API MLflow
EXPOSE 5001

# Jalankan server MLflow ketika container dinyalakan
CMD ["mlflow", "models", "serve", "-m", "/app/model", "-h", "0.0.0.0", "-p", "5001", "--no-conda"]