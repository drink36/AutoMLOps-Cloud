# image base
FROM pytorch/pytorch:2.6.0-cuda11.8-cudnn9-runtime

# 2. set environment variables and working directory
ARG DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1 \
    SM_MODEL_DIR=/opt/ml/model \
    WORKDIR=/opt/ml/code \
    MODE=inference \
    SAGEMAKER_PROGRAM=inference.py
WORKDIR ${WORKDIR}
RUN mkdir -p ${SM_MODEL_DIR}

# 3. install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
      gcc build-essential python3-venv \
    && rm -rf /var/lib/apt/lists/*

# 4. copy requirements and install python dependencies
COPY requirements-common.txt requirements-train.txt requirements-infer.txt ./
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements-common.txt && \
    pip install --no-cache-dir -r requirements-train.txt && \
    pip install --no-cache-dir -r requirements-infer.txt

# 5. copy code and entrypoint script
COPY . ${WORKDIR}
COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

# 6. expose port 8080 for inference
EXPOSE 8080

# 7. set entrypoint
ENTRYPOINT ["entrypoint.sh"]
