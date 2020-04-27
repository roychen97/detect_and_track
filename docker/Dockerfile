FROM tensorflow/tensorflow:1.15.0-gpu-py3

RUN apt-get update && \
    apt-get install -y \
    cmake \
    libhdf5-dev \
    g++ \
    vim \
    wget \
    git \
    yasm \
    curl \
    locate \
    libgtk2.0-dev \
    pkg-config \
    unzip \
    build-essential \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    python-dev \
    python-numpy \
    libtbb2 \
    libtbb-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libavformat-dev \
    libpq-dev \
    libcurl4-openssl-dev \
    libgflags-dev \
    libssl-dev
   
RUN add-apt-repository "deb http://security.ubuntu.com/ubuntu xenial-security main"
RUN apt update
RUN apt install libjasper1 libjasper-dev

# ============= OpenCV =============
# OpenCV
# [Warning] - opencv-python does not include Video I/O (Can not open Webcam)
# Build opencv from source

ARG opencv_version=3.4.1

ENV OPENCV_DIR /opt/opencv

RUN mkdir -p $OPENCV_DIR

WORKDIR $OPENCV_DIR

RUN wget https://github.com/Itseez/opencv/archive/${opencv_version}.zip -O opencv-${opencv_version}.zip
RUN wget https://github.com/Itseez/opencv_contrib/archive/${opencv_version}.zip -O opencv_contrib-${opencv_version}.zip

RUN unzip opencv-${opencv_version}.zip && \
    unzip opencv_contrib-${opencv_version}.zip && \
    cd opencv-${opencv_version} &&\
    mkdir build && cd build && \
    cmake -D CMAKE_BUILD_TYPE=Release \
    -D OPENCV_EXTRA_MODULES_PATH=/opt/opencv/opencv_contrib-${opencv_version}/modules \
    -D CMAKE_INSTALL_PREFIX=$(python -c "import sys; print(sys.prefix)") \
    -D WITH_CUDA=OFF \
    ..

RUN cd opencv-${opencv_version}/build && \
    make -j"$(nproc)" && \
    make install && \
    rm $OPENCV_DIR/opencv-${opencv_version}.zip && \
    rm $OPENCV_DIR/opencv_contrib-${opencv_version}.zip

RUN pip3 install matplotlib
RUN pip3 install scipy
RUN pip3 install Pillow
RUN pip3 install easydict
RUN sed -i '446c\ \ \ \ \ \ \ \ \ \ \ \ \ \ \ fused=False,' /usr/local/lib/python3.6/dist-packages/tensorflow_core/contrib/layers/python/layers/layers.py
RUN sed -i '1428c\ \ \ \ inv = math_ops.rsqrt(math_ops.add(variance,variance_epsilon))' /usr/local/lib/python3.6/dist-packages/tensorflow_core/python/ops/nn_impl.py
RUN sed -i '1433c\ \ \ \ return math_ops.add( x * inv , offset - mean * inv if offset is not None else -mean * inv)' /usr/local/lib/python3.6/dist-packages/tensorflow_core/python/ops/nn_impl.py
RUN sed -i '1434c\ \ \ \ #' /usr/local/lib/python3.6/dist-packages/tensorflow_core/python/ops/nn_impl.py
WORKDIR /workspace
RUN chmod -R a+w /workspace

