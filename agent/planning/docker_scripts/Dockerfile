FROM ubuntu:latest

RUN /bin/bash -c "echo 'deb http://archive.ubuntu.com/ubuntu trusty-backports main restricted universe multiverse' >> /etc/apt/sources.list"
RUN apt update
RUN apt install -y -t trusty-backports flex=2.5.39-8~ubuntu14.04.1
RUN apt-mark hold flex

RUN apt-get install -y build-essential bison libc6-dev-i386 gcc-multilib g++-multilib git

WORKDIR /home/

RUN git clone --single-branch --branch master-x64 https://github.com/gdellapenna/UPMurphi.git

WORKDIR /home/UPMurphi/src
RUN make

WORKDIR /home/UPMurphi/ex/science_birds

COPY ./run_script.sh /home/UPMurphi/ex/science_birds/
COPY ./sb_domain.pddl /home/UPMurphi/ex/science_birds/
COPY ./sb_prob.pddl /home/UPMurphi/ex/science_birds/

ENTRYPOINT ["/bin/bash", "run_script.sh"]