FROM continuumio/miniconda3:4.8.2

WORKDIR /hydra

COPY . .

RUN conda env create --file environment.yml

ENV PATH /opt/conda/envs/hydra/bin:$PATH
ENV PYTHONPATH ".:./worlds/science_birds_interface/"
