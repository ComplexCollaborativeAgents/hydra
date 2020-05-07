FROM continuumio/miniconda3:4.8.2

WORKDIR /hydra

COPY environment.yml environment.yml

RUN conda env create --file environment.yml
SHELL ["conda", "run", "-n", "hydra", "/bin/bash", "-c"]

COPY . .
RUN pip install -e .

CMD ["conda", "run", "-n", "hydra", "python", "runners/simple.py"]
