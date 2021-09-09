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
2. Install a specific version of Java8 (not needed if already installed)  
   2.1. 
   ``` 
   sudo wget -qO - https://adoptopenjdk.jfrog.io/adoptopenjdk/api/gpg/key/public |  apt-key add -
   ```
   2.2.
   ```
   sudo add-apt-repository --yes https://adoptopenjdk.jfrog.io/adoptopenjdk/deb/
   ```
   2.3.
   ```
   apt-get update
   ```
   2.4.
   ```
   apt-get install adoptopenjdk-8-hotspot=8u232-b09-2 -V -y
   ```
3. As a root user (sudo), run the ```setup_polycraft.sh``` bash script under the binary directory to install linux dependencies.
4. Note that the Polycraft domain requires a different version of Java than that of the Science Birds domain.  If using Ubuntu, you can use the `sudo update-alternatives --config java` command to select the Java version that corresponds with the domain that you want to run in.
5. Create the hydra python environment (same as Science Birds above - again, not needed if already installed)
6. Download the polycraft level files from https://gitlab-external.parc.com/hydra/hydra/-/wikis/Polycraft-Level-files and put them in a known directory.

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
