# HYDRA
## Setting up Science Birds
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
## Setting up Polycraft
1. Unzip the Polycraft repository in the binary directory.  Currently, we only support Polycraft for Ubuntu.
2. Install Java13 (not needed if already installed)
3. As a root user (sudo), run the ```setup_polycraft.sh``` bash script under the binary directory to install linux dependencies.
4. Create the hydra python environment (same as Science Birds above - again, not needed if already installed)

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
