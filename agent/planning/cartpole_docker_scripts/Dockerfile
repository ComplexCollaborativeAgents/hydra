FROM ubuntu:latest

RUN /bin/bash -c "echo 'deb http://archive.ubuntu.com/ubuntu trusty-backports main restricted universe multiverse' >> /etc/apt/sources.list"
RUN apt update
RUN apt-get update && apt-get upgrade -y
RUN apt install -y -t trusty-backports flex=2.5.39-8~ubuntu14.04.1
RUN apt-mark hold flex

RUN apt-get install -y build-essential bison libc6-dev-i386 gcc-multilib g++-multilib git unzip

WORKDIR /home/

COPY ./UPMurphi-master-x64-DFS.zip /home/

RUN unzip UPMurphi-master-x64-DFS.zip

WORKDIR /home/UPMurphi-DFS/src
RUN make

WORKDIR /home/UPMurphi-DFS/ex/cartpole

COPY ./run_script.sh /home/UPMurphi-DFS/ex/cartpole/
COPY ./cartpole_domain.pddl /home/UPMurphi-DFS/ex/cartpole/
COPY ./cartpole_prob.pddl /home/UPMurphi-DFS/ex/cartpole/

ENTRYPOINT ["/bin/bash", "run_script.sh"]