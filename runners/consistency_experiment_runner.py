import sklearn.metrics as metrics
import numpy as np
from agent.repair.sb_repair import ScienceBirdsConsistencyEstimator, BirdLocationConsistencyEstimator, BlockNotDeadConsistencyEstimator
import enum
import pathlib
from typing import List
import matplotlib.pyplot as plt
from agent.consistency.consistency_estimator import check_obs_consistency
from agent.consistency.fast_pddl_simulator import *

from state_prediction.anomaly_detector_fc_multichannel import FocusedSBAnomalyDetector

logging.basicConfig(format='%(name)s - %(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("run_consistency_check_experiments")
logger.setLevel(logging.INFO)


SB_OBS_PATH = pathlib.Path(settings.ROOT_PATH) / 'data' / 'science_birds' / 'consistency'
STATS_FILE_PATH = pathlib.Path(__file__).parent.absolute()
STATS_FILE_TEMPLATE = "consistency-results-{exp_name}.csv"

SAMPLES = 100

DELIMITER = ','

class ConsistencyCheckerType(enum.Enum):
    ScienceBirds = ScienceBirdsConsistencyEstimator
    BirdLocation = BirdLocationConsistencyEstimator
    BlockNotDead = BlockNotDeadConsistencyEstimator


''' Run a consistency check experiment, in which we run the consistency checker on novelty and non-novelty levesl and measure consistency '''
def run_consistency_stats(consistency_checker_types: List[ConsistencyCheckerType],
                          seed: int = None,
                          samples: int = SAMPLES,
                          observations_path: pathlib.Path = SB_OBS_PATH,
                          results_file_template: str = STATS_FILE_TEMPLATE):

    if seed is not None:
        random.seed(seed)

    results_file_name = results_file_template.format(exp_name="1")
    print(STATS_FILE_PATH / results_file_name)
    outfile = (STATS_FILE_PATH / results_file_name).open("w")
    outfile.write("{}\n".format(DELIMITER.join(["Checker", "Novel", "Observation", "Consistency"])))


    print("Novel cases: Running PDDL consistency checkers")
    is_novel = "True"
    obs_path = pathlib.Path(observations_path) / 'novel'
    obs_files = list(obs_path.glob('*_observation.p'))
    obs_files = random.sample(obs_files, min(len(obs_files), samples))

    for consistency_checker_type in consistency_checker_types:
        consistency_checker = consistency_checker_type.value()
        exp_name = DELIMITER.join((consistency_checker_type.name, is_novel))
        run_experiments(exp_name,
                        consistency_checker,
                        obs_files,
                        outfile)
    print("Running UPenn detector")
    detector = FocusedSBAnomalyDetector()
    exp_name = DELIMITER.join(("UPenn", is_novel))
    run_upenn_detector(exp_name, detector, obs_files, outfile)

    print("Non-novel cases: Running PDDL consistency checkers")
    is_novel = "False"
    obs_path = pathlib.Path(observations_path) / 'non_novel'
    obs_files = list(obs_path.glob('*_observation.p'))
    obs_files = random.sample(obs_files, samples)
    for consistency_checker_type in consistency_checker_types:
        print("Running consistency checker {}".format(consistency_checker))
        consistency_checker = consistency_checker_type.value()
        checker_name = consistency_checker_type.name
        exp_name = DELIMITER.join((checker_name, is_novel))
        run_experiments(exp_name,
                        consistency_checker,
                        obs_files,
                        outfile)

    print("Running UPenn detector")
    detector = FocusedSBAnomalyDetector()
    exp_name = DELIMITER.join(("UPenn", is_novel))
    run_upenn_detector(exp_name, detector, obs_files, outfile)
    outfile.close()


def run_experiments(exp_name, consistency_checker, obs_files, outfile):
    ''' Runs the given consistency checker on the set of given set of files and output the results to the results file'''

    simulator = CachingPddlPlusSimulator()
    meta_model = ScienceBirdsMetaModel()
    for obs_file in obs_files:
        obs = pickle.load(open(obs_file, "rb"))
        consistency_value = check_obs_consistency(obs, meta_model, consistency_checker, simulator=simulator)
        results_line = "{}\n".format(DELIMITER.join((exp_name , obs_file.name, str(consistency_value))))
        logger.info(results_line)
        outfile.write(results_line)
        outfile.flush()


def run_upenn_detector(exp_name, detector:FocusedSBAnomalyDetector, obs_files, outfile):
    ''' Runs UPenn's detector on the given obs_files and output appropriate results file '''

    for obs_file in obs_files:
        obs = pickle.load(open(obs_file, "rb"))

        novelties, prob = detector.detect(obs)

        results_line = "{}\n".format(DELIMITER.join((exp_name , obs_file.name, "{:.5f}".format(prob))))
        logger.info(results_line)
        outfile.write(results_line)
        outfile.flush()

def analyze_results(results_file: pathlib.Path, alg_name = ""):
    ''' Compute the ROC curve of the results file '''

    lines = open(results_file,"r").readlines()
    headers = [value.strip() for value in lines[0].split(DELIMITER)]
    novel_idx = -1
    consistency_idx = -1
    for i in range(len(headers)):
        if headers[i]=="Novel":
            novel_idx = i
        if headers[i]=="Consistency":
            consistency_idx = i
    dataset = list()
    y = list()
    pred = list()
    for line in lines[1:]:
        record = [value.strip() for value in line.split(DELIMITER)]
        if record[novel_idx]=="True":
            record[novel_idx]=1.0
        elif record[novel_idx]=="False":
            record[novel_idx]=0.0
        else:
            raise ValueError("Novelty value is {}".format(record[novel_idx]))
        record[consistency_idx]=float(record[consistency_idx])
        dataset.append(record)

        y.append(record[novel_idx])
        pred.append(record[consistency_idx])

    y = np.array(y)
    pred = np.array(pred)

    fpr, tpr, thresholds = metrics.roc_curve(y, pred, pos_label=1.0)
    roc_auc = metrics.auc(fpr, tpr)

    plt.title('Receiver Operating Characteristic - {}'.format(alg_name))
    plt.plot(fpr, tpr, label='{} AUC = {:.2f}'.format(alg_name, roc_auc))
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()

if __name__ == '__main__':
    # algs_to_test = [ConsistencyCheckerType.ScienceBirds,
    #                 ConsistencyCheckerType.BirdLocation,
    #                 ConsistencyCheckerType.BlockNotDead]

    algs_to_test = [ConsistencyCheckerType.ScienceBirds]
    results_file_template = STATS_FILE_TEMPLATE
    run_consistency_stats(algs_to_test, results_file_template=results_file_template, samples=50, seed=0)
    # logger.info("Experiment Done! Now analyzing...")
    # for alg in algs_to_test:
    #     results_file_name = results_file_template.format(exp_name=alg.name)
    #     results_file = (STATS_FILE_PATH / results_file_name)
    #     print("Writing results to file {}".format(results_file_name))
    #
    #
    #     analyze_results(results_file, alg.name)
    logger.info("Analysis Done!!")