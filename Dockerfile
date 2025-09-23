# Simple single-container image with Python, PySpark, and JupyterLab
FROM openjdk:11-jre-slim

# Avoid interactive tzdata
ENV DEBIAN_FRONTEND=noninteractive

# Install Python and basic tools
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install Spark
ENV SPARK_VERSION=3.4.1
ENV HADOOP_VERSION=3
RUN wget -q "https://archive.apache.org/dist/spark/spark-${SPARK_VERSION}/spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz" \
    && tar -xzf "spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz" \
    && mv "spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}" /opt/spark \
    && rm "spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz"

ENV SPARK_HOME=/opt/spark
ENV PATH=$PATH:$SPARK_HOME/bin:$SPARK_HOME/sbin
ENV PYTHONPATH=$SPARK_HOME/python:$SPARK_HOME/python/lib/py4j-0.10.9.7-src.zip

# Install Python deps
RUN pip3 install --no-cache-dir \
    pyspark==3.4.1 \
    pandas==2.0.3 \
    numpy==1.24.3 \
    jupyterlab==4.0.5

# Default workdir
WORKDIR /workspace

# Expose Jupyter port
EXPOSE 8888

# Default command is overridden by docker-compose
CMD ["bash"]
