This file includes information about the scripts in the utils directory. There are two purposes of the code here.
    1) Generating visual object perception data and training the visual classifier
    2) Generating processed observation data for UPenn's representation learning approach

* Observation generation code
** Generate_observations.py
The main function has a single parameter config. The most useful settings for this parameter are all_level_0_novelties.xml,
along with each specific level of novelites. When the script is run, two sets of files are created. In the main project directory,
    - known_objects.csv: These are the unprocessed observations for all the known objects in the scene.
    - unknown_objects.csv: These are the unprocessed observations for all the novel objects in the scene.
We get the labels by have the SBDEV mode set to true, that is how we know which objects are novel or not.



CONCERN: Does the --random flag work?

* Object classification code

