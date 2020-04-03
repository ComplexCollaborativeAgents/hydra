FROM ubuntu:latest

RUN /bin/bash -c "echo 'deb http://archive.ubuntu.com/ubuntu trusty-backports main restricted universe multiverse' >> /etc/apt/sources.list"
RUN apt update
RUN apt install -y -t trusty-backports flex=2.5.39-8~ubuntu14.04.1
RUN apt-mark hold flex

RUN apt-get update -y && \
  apt-get install --no-install-recommends -y -q \
  # install cmake
  cmake \
  # install make g++
  make g++ \
  # install clang-format
  clang-format-3.9 \
  # install doxygen
  doxygen \
  graphviz \
  # install mingw
  mingw-w64 \
  # install zip
  zip unzip \
  && \
  apt-get clean && \
  rm /var/lib/apt/lists/*_*

WORKDIR /home/VAL/

COPY ./val-dev-linux.tar.gz /home/VAL/
RUN tar -xvzf val-dev-linux.tar.gz

COPY ./run_val.sh /home/VAL/
COPY ./sb_domain.pddl /home/VAL/
COPY ./sb_prob.pddl /home/VAL/
COPY ./docker_plan_trace.txt /home/VAL/

ENTRYPOINT ["/bin/bash", "run_val.sh"]