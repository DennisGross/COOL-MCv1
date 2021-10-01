FROM python:3.8


RUN apt-get update
RUN apt-get -y install build-essential git cmake libboost-all-dev libcln-dev libgmp-dev libginac-dev automake libglpk-dev libhwloc-dev libz3-dev libxerces-c-dev libeigen3-dev


RUN git clone https://github.com/moves-rwth/storm.git
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

RUN git clone https://github.com/moves-rwth/stormpy.git
WORKDIR /stormpy
RUN python3.8 setup.py build_ext --storm-dir /storm/build/ --jobs 1 develop

WORKDIR /
COPY cool_mc.py .
COPY src src/
COPY requirements.txt .
COPY example_1.sh .
COPY example_2.sh .
COPY example_3.sh .
COPY experiments_consensus.sh .
COPY experiments_frozen_lake.sh .
COPY experiments_modified_qcomp.sh .
COPY experiments_qcomp.sh .
COPY experiments_resource_gathering.sh .
COPY experiments_smart_grid.sh .
COPY experiments_taxi.sh .

RUN apt-get install graphviz -y
RUN pip3.8 install -r requirements.txt
ENTRYPOINT /bin/bash
