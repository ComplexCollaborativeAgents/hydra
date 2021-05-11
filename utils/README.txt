This file includes information about the scripts in the utils directory. This is file is organized by what one is trying to do.

* Create Observations for UPenn's Training

The code that creates the observations for UPenn is in generate_observations.py. The main function takes a config file,
which specifies the science birds problems to load. UPenn needs non-novelty observations to train on all_level_0_novelties.xml.
Then they also need novelty observations to create thresholds, e.g., 120_sample_novelties.xml. Then there are three settings that are overridden.
    - DEBUG should be True to ensure that the observations are written out to files, as done in handle_game_playing
    - SB_DEV_MODE should be False, when this flag is true, we are given the class names of every object and there is no positional noise
    - NO_PLANNING should be True, when this flag is true, HYDRA just shoots at a random pig.
UPenn would like to have half the observations be from shots at random pigs in the scene and half the shots be random shots in the scene. To
generate this, there is a script at the root directory called generate_obs.sh that will use the --random flag to generate the random shots.

* Retraining the Object Classifier
The object classifier is a logistic regression model.

https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html

The model is in data/science_birds/perception/logreg.p.

I believe that the files model and target_class are not used anymore.

If you want to retrain the classifier, you need a training and test set. These are stored in
object_classification.tar.gz. When unzipped, there are two CSV's that are used with in assess_classification.py: object_class_level_0.csv
and object_class_level_1.csv. The main function calls train_classifier() which trains the logistic regression model
using object_class_level_0.csv, and then evaluates it on object_class_level_1.csv. The final output should be 0, which means
every object was classified correctly in the test set.

Note: We don't try to classify directly to the SB_DEV_MODE classes. We abstract their 51 classes into roughtly 11 using
the type_to_class() function.

* Generating Object Classification Data
WARNING: SB_DEV_MODE IS ONLY PROVIDING THE SLINGSHOT CLASS IN VERSION 0.4.1. I have posted the following slack message:
https://ta1angrybirds.slack.com/archives/CRWALP52N/p1618454828009600

To generate new object_class_level_0.csv and object_class_level_1.csv files. You simply need to run generate_observations.py
with the following settings.
    - DEBUG set to True will write out a file called object_class.csv.
    - SB_DEV_MODE set to True will provide the class of each object.
    - NO_PLANNING set to True as we don't want to waste time planning.

To generate object_class_level_0.csv, run generate_observations with config set to all_level_0_novelties.xml and then
copy the object_class.csv file into the object_class_level_0.csv in data/science_birds_perception/. Then do the same for
all_level_1_novelties.xml and copy the new object_class.csv into object_class_level_1.csv.

* Generating New Config Files
In assess_classification.py, there is generate_config_files() which will create a config file based on a given directory.