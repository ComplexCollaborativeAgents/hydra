# HYDRA
## Setting up
1. Unzip the science birds executable for your systems in the binary directory
   - Look at .gitlab-ci.yml for detailed instructions
2. Install Java13 (for additional debugging of science birds issues look at the science birds repo below)
```https://gitlab.com/aibirds/sciencebirdsframework```
3. create the hydra python environment
   - ```conda env create -f environment.yml```
   - ```conda activate hydra```
   - `pip install -e .`
4. run the tests from HYDRA directory
```
pytest
```

## Building Docker

```
# Log in to the GitLab Docker registry.
$ docker login registry.gitlab-external.parc.com:8443

# Build Docker image for latest commit.
$ docker build -t registry.gitlab-external.parc.com:8443/hydra/experiment:$(git rev-parse --short HEAD) .

# Tag the most recent Docker image with `latest`.
$ docker tag registry.gitlab-external.parc.com:8443/hydra/experiment:$(git rev-parse --short HEAD) registry.gitlab-external.parc.com:8443/hydra/experiment:latest

# Push versioned and latest image to the registry.
$ docker push registry.gitlab-external.parc.com:8443/hydra/experiment:$(git rev-parse --short HEAD)

$ docker push registry.gitlab-external.parc.com:8443/hydra/experiment:latest
```
