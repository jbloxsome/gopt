FROM ubuntu:latest

RUN apt-get -y update
RUN apt-get -y install apt-utils wget git gcc build-essential
ENV DEBIAN_FRONTEND=noninteractive
RUN ln -fs /usr/share/zoneinfo/Europe/Moscow /etc/localtime
RUN apt-get install -y tzdata
RUN dpkg-reconfigure --frontend noninteractive tzdata
RUN apt-get -y install wget sudo unzip git intel-mkl-full cmake golang
ENV LIBTORCH_PATH="/usr/local/lib"
ENV GOTCH_LIBTORCH="$LIBTORCH_PATH/libtorch"
ENV LIBRARY_PATH="$LIBRARY_PATH:$GOTCH_LIBTORCH/lib:/usr/lib/x86_64-linux-gnu/mkl"
ENV CPATH="$CPATH:$GOTCH_LIBTORCH/lib:$GOTCH_LIBTORCH/include:$GOTCH_LIBTORCH/include/torch/csrc/api/include:/usr/include"
ENV GOPATH="/root/go"

RUN export LIBTORCH_ZIP="libtorch-cxx11-abi-shared-with-deps-1.11.0%2Bcpu.zip" \
    && wget  -q --show-progress --progress=bar:force:noscroll  -O "/tmp/$LIBTORCH_ZIP" "https://download.pytorch.org/libtorch/cpu/$LIBTORCH_ZIP" \
    && unzip "/tmp/$LIBTORCH_ZIP" -d $LIBTORCH_PATH \
    && rm "/tmp/$LIBTORCH_ZIP" \
    && ldconfig

ENV GOTCH_LIBTORCH="/usr/local/lib/libtorch"
ENV LIBRARY_PATH="$LIBRARY_PATH:$GOTCH_LIBTORCH/lib"
ENV CPATH="$CPATH:$GOTCH_LIBTORCH/lib:$GOTCH_LIBTORCH/include:$GOTCH_LIBTORCH/include/torch/csrc/api/include"
ENV LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$GOTCH_LIBTORCH/lib"

COPY setup-gotch.sh setup-gotch.sh
RUN chmod +x setup-gotch.sh
RUN export CUDA_VER=cpu && export GOTCH_VER=v0.7.0 && bash setup-gotch.sh

COPY src /app

WORKDIR /app

CMD [ "go", "run", "." ]