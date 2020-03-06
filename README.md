# HYDRA
## Setting up
1. Unzip the science birds executable for your systems in the binary directory
   - Look at .gitlab-ci.yml for detailed instructions
2. Install Java13 (for additional debugging of science birds issues look at the science birds repo below)
```https://gitlab.com/aibirds/sciencebirdsframework```
3. create the hydra python environment
   - ```conda env create -f environment.yml```
   - ```conda activate hydra```
4. run the tests from HYDRA directory
```
pytest
```
