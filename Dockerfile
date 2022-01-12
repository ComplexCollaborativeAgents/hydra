FROM continuumio/miniconda3:4.8.2

# Needed for pybullet
RUN apt-get --allow-releaseinfo-change update && apt-get install -y build-essential

WORKDIR /hydra

COPY . .

RUN conda env create --file environment.yml && conda clean -ay

ENV PATH /opt/conda/envs/hydra/bin:$PATH
ENV PYTHONPATH ".:./worlds/science_birds_interface/"
