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
2. Install a specific version of Java 8 (adoptopenjdk-8-hotspot)  
   2.1 sudo wget -qO - https://adoptopenjdk.jfrog.io/adoptopenjdk/api/gpg/key/public | sudo apt-key add -  
   2.2 sudo add-apt-repository --yes https://adoptopenjdk.jfrog.io/adoptopenjdk/deb/  
   2.3 sudo apt-get update  
   2.4 apt-get install adoptopenjdk-8-hotspot -V -y  
   2.5 sudo sed -i -e '/^assistive_technologies=/s/^/#/' /etc/java-*-openjdk/accessibility.properties  
3. Change directories to `bin` and run `./setup_polycraft.sh`.  This will install a specific version of Java 8 in addition to any other linux dependencies.  
4. Note that the Polycraft domain requires a different version of Java than that of the Science Birds domain.  If using Ubuntu, you can use the `sudo update-alternatives --config java` command to select the Java version that corresponds with the domain that you want to run in.  If the Polycraft Hydra Agent hangs after starting up, check to make sure that you are running Java 8 (version adoptopenjdk8-hotspot). 
5. Create the hydra python environment (same as Science Birds above - again, not needed if already installed)
6. Polycraft will need to do a one time setup.  Navigate to `bin/pal` and run `xvfb-run -s '-screen 0 1280x1024x24' ./gradlew --no-daemon --stacktrace runclient`.  This will run Polycraft independently in headless mode.  Gradle will install any dependencies that the java runtime needs, and eventually a message will appear in the log output `[EXP] game initialization completed`, which signifies that Polycraft is ready to use.  Exit out of the application.
7. Download the polycraft level files from https://gitlab-external.parc.com/hydra/hydra/-/wikis/Polycraft-Level-files and unzip them in a known directory.  

   7.1. Level file naming convention (using POGO_L01_T01_S01_X0010_A_U0002_V2 as an example)
      |Segment|Description|
      |---|---|
      | POGO | task name |  
      | LXX | novelty level |  
      | TXX | novelty type |  
      | SXX | novelty subtype |  
      | X0010 | The X doesn't mean anything, but the number that follows is the number of games within the trial |  
      | A | Difficulty rating, where E = Easy, M = Medium, H = Hard, A = All/Mixed difficulty.  Meaningless in pre-novelty trial instances |  
      | UXXXX | U means that the agent will not be informed of novelty presence. K means that novelty presence will be given.  The number that follows is the episode number in which novelty is introduced.  The count starts at 0. |  
      | VX | Variant of the same trial (different levels, same trial structure) |  
      
   7.2. Inside the overall folder, there exists subsets of novelty levels and types. The lowest level of the directory contain more zipped files, which must be unzipped before being used.  
   7.3. Create or add to your `settings/local_settings.py` file and update the `POLYCRAFT_LEVEL_DIR` variable with an absolute path to the level files directory.  
   7.4. Note that within bin/pal there exists a `pogo_100_PN` directory that contains sample levels that can be referenced without downloading the level files.  
   7.5. Additional note: The Polycraft application is unable to run non-headless when used in the `polycraft_dispatcher.py` or in `bin/pal/PolycraftAIGym/LaunchTournament.py` or any other python script that runs it via subprocess on its own.  A workaround would be to run the application in a separate command line window using `./gradlew --no-daemon --stacktrace runclient` (also can be found in settings.POLYCRAFT_SERVER_CMD) from within the `bin/pal` directory, then setting up another Python script with a `Polycraft` world object with the optional `launch` boolean set to False. The caveat with this is that you will need to time the agent start when the Polycraft application is fully initialized, or the process will crash.  

## Testing Polycraft
   There exists a `test_polycraft.py` file within `tests/` that tests if the installation is working by loading a level and running a set of actions upon it.  To run it:  
   1. `cd tests`
   2. `pytest -ra test_polycraft.py::test_polycraft`


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
