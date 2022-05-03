import pathlib
import enum
import settings
import pandas
import glob
import json
import re
import numpy
from scipy.stats import pearsonr
from sklearn.linear_model import LogisticRegression


SB_NOVELTY_DATA_PATH = pathlib.Path(settings.ROOT_PATH) / 'data' / 'science_birds' / 'novelty_detection' / 'dataset_May2022'
NON_NOVELTY_LEVELs = [0]
NOVELTY_LEVELS = [22, 23, 24, 25]

class ColumnName(enum.Enum):
    LEVEL = 'level',
    TYPE = 'type',
    LEVEL_TYPE = 'level_type'
    HAS_NOVEL_OBJECT = 'has_novel_object',
    MAX_REWARD_DIFFERENCE = 'max_reward_difference',
    AVG_REWARD_DIFFERENCE = 'avg_reward_difference',
    MAX_PDDL_INCONSISTENCY = 'max_pddl_inconsistency',
    AVG_PDDL_INCONSISTENCY = 'avg_pddl_inconsistency',
   # HYDRA_NOVELTY_DETECTED = 'hydra_novelty_detected',
    GROUND_TRUTH = 'ground_truth',
    NUM_OBJECTS = 'num_objects',
    PASS = 'pass'


def generate_dataset_from_json():
    dataframe = pandas.DataFrame(columns=[
        ColumnName.LEVEL_TYPE,
        ColumnName.LEVEL,
        ColumnName.TYPE,
        ColumnName.NUM_OBJECTS,
        ColumnName.HAS_NOVEL_OBJECT,
        ColumnName.MAX_REWARD_DIFFERENCE,
        ColumnName.AVG_REWARD_DIFFERENCE,
        ColumnName.MAX_PDDL_INCONSISTENCY,
        ColumnName.AVG_PDDL_INCONSISTENCY,
        #ColumnName.HYDRA_NOVELTY_DETECTED,
        ColumnName.GROUND_TRUTH
    ])

    datafiles = glob.glob("{}/*novelty*.json".format(SB_NOVELTY_DATA_PATH))
    print(datafiles)

    for file in datafiles:
        with open(file, 'r') as f:
            filedata = json.load(f)
            leveldata = filedata['levels']
            numbers = re.findall(r'\d+', file)
            print("Processing file {}".format(file))
            print("numbers {}".format(numbers))
            for episode in leveldata:
                if True in episode['unknown_object']:
                    has_unknown_object = 1
                else:
                    has_unknown_object = 0
                # if episode['novelty_detection']:
                #     hydra_detected_novelty = 1
                # else:
                #     hydra_detected_novelty = 0

                if episode['status'] == "Pass":
                    status = 1
                else:
                    status = 0


                if all(v is None for v in episode['pddl_novelty_likelihood']):
                    #max_pddl_inconsistency = numpy.NAN
                    #avg_pddl_inconsistency = numpy.NAN
                    max_pddl_inconsistency = 1000
                    avg_pddl_inconsistency = 1000
                else:
                    max_pddl_inconsistency = numpy.nanmax(episode['pddl_novelty_likelihood'])
                    avg_pddl_inconsistency = numpy.nanmean(episode['pddl_novelty_likelihood'])

                if numbers[1] is '0':
                    ground_truth = 0
                else:
                    ground_truth = 1

                if len(episode['reward_estimator_likelihood']) == 0:
                    max_reward_difference = 0
                    avg_reward_difference = 0
                else:
                    max_reward_difference = numpy.nanmax(episode['reward_estimator_likelihood'])
                    avg_reward_difference = numpy.nanmean(episode['reward_estimator_likelihood'])

                line = {
                    ColumnName.LEVEL: numbers[1],
                    ColumnName.TYPE: numbers[2],
                    ColumnName.LEVEL_TYPE: float("{}.{}".format(numbers[1], numbers[2])),
                    ColumnName.NUM_OBJECTS: episode['objects'],
                    ColumnName.HAS_NOVEL_OBJECT: has_unknown_object,
                    ColumnName.MAX_REWARD_DIFFERENCE: max_reward_difference,
                    ColumnName.AVG_REWARD_DIFFERENCE: avg_reward_difference,
                    ColumnName.MAX_PDDL_INCONSISTENCY: max_pddl_inconsistency,
                    ColumnName.AVG_PDDL_INCONSISTENCY: avg_pddl_inconsistency,
                    ColumnName.GROUND_TRUTH: ground_truth,
                    ColumnName.PASS: int(status)
                }

                #print(line)

                dataframe = dataframe.append(line, ignore_index=True)

    dataframe = dataframe.astype({ColumnName.PASS: 'int64'})
    print(dataframe.dtypes)
    dataframe.to_csv("{}/ensemble_learning_simple_levels_may2022.csv".format(SB_NOVELTY_DATA_PATH))
    return(dataframe)


if __name__ == '__main__':
    dataframe = generate_dataset_from_json()

