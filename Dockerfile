FROM python:3.8


RUN apt-get update
RUN apt-get -y install build-essential git cmake libboost-all-dev libcln-dev libgmp-dev libginac-dev automake libglpk-dev libhwloc-dev
RUN apt-get -y install build-essential git cmake libboost-all-dev libcln-dev libgmp-dev libginac-dev automake libglpk-dev libhwloc-dev libz3-dev libxerces-c-dev libeigen3-dev

RUN git clone --branch prismlang-sim https://github.com/sjunges/storm.git
WORKDIR /storm
RUN mkdir build
WORKDIR /storm/build
RUN cmake ..
RUN make -j 1
WORKDIR /
RUN apt-get update -qq && apt-get install -y --no-install-recommends \
    maven \
    uuid-dev \
    python3 \
    virtualenv

RUN git clone https://github.com/moves-rwth/pycarl.git
WORKDIR /pycarl
RUN python3 setup.py build_ext --jobs 1 develop

WORKDIR /
RUN git clone --branch prismlang-sim https://github.com/sjunges/stormpy.git
WORKDIR /stormpy
RUN python3.8 setup.py build_ext --storm-dir /storm/build/ --jobs 1 develop

WORKDIR /
COPY cool_mc.py .
COPY src src/
COPY requirements.txt .
COPY example_1.sh .
RUN pip3.8 install -r requirements.txt
RUN pip3.8 install tensorflow_datasets
RUN pip3.8 install tensorflow_text
RUN pip3.8 install matplotlib
ENTRYPOINT /bin/bash