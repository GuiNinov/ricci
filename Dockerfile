FROM rust

WORKDIR /app
RUN apt update &&\
    rm -rf ~/.cache &&\
    apt clean all &&\
    apt install -y cmake &&\
    apt install -y clang


# install libtorch=2.3.0
# https://pytorch.org/get-started/locally/
RUN wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.3.0%2Bcpu.zip -O libtorch.zip
RUN unzip -o libtorch.zip
ENV LIBTORCH /app/libtorch
ENV LD_LIBRARY_PATH /app/libtorch/lib:$LD_LIBRARY_PATH

# file
COPY ./Cargo.toml /app/Cargo.toml
COPY ./src /app/src
COPY ./assets /app/assets