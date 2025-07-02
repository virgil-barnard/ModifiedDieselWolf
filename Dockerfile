# Use official PyTorch image with CUDA runtime for GPU support
ARG CUDA_VERSION=12.1.1
ARG CUDNN_VERSION=8.9.2
FROM nvcr.io/nvidia/pytorch:25.04-py3
ENV CUDA_VERSION=${CUDA_VERSION} \
    CUDNN_VERSION=${CUDNN_VERSION}
# https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN chmod +x entrypoint.sh
EXPOSE 8888
ENTRYPOINT ["./entrypoint.sh"]
