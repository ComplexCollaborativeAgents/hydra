FROM continuumio/miniconda3:4.8.2

WORKDIR /hydra

SHELL ["/bin/bash", "-c"]

COPY environment.yml environment.yml

RUN conda env create --file environment.yml
RUN source activate hydra

COPY . .
RUN pip install .

CMD ["python", ""]