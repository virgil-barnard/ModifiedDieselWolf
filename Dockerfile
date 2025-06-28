# Use official PyTorch image with CUDA runtime for GPU support
FROM nvcr.io/nvidia/pytorch:25.04-py3
# https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN chmod +x entrypoint.sh
EXPOSE 8888
ENTRYPOINT ["./entrypoint.sh"]