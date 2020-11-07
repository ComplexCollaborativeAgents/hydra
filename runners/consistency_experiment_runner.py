import sklearn.metrics as metrics
import numpy as np
from agent.consistency.sb_repair import ScienceBirdsConsistencyEstimator, BirdLocationConsistencyEstimator, BlockNotDeadConsistencyEstimator
import enum
import pathlib
from typing import List
import matplotlib.pyplot as plt
from agent.consistency.consistency_estimator import check_obs_consistency, ConsistencyEstimator
from agent.consistency.fast_pddl_simulator import *


logging.basicConfig(format='%(name)s - %(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("run_consistency_check_experiments")
logger.setLevel(logging.INFO)


SB_OBS_PATH = pathlib.Path(settings.ROOT_PATH) / 'data' / 'science_birds' / 'consistency' / 'dynamics'
STATS_FILE_PATH = pathlib.Path(__file__).parent.absolute()
STATS_FILE_TEMPLATE = "consistency-results-{alg_name}.csv"

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

    is_novel = "True"
    obs_path = pathlib.Path(observations_path) / 'novel'
    obs_files = list(obs_path.glob('*_observation.p'))
    obs_files = random.sample(obs_files, samples)

    for consistency_checker_type in consistency_checker_types:
        results_file_name = results_file_template.format(alg_name=consistency_checker_type.name)
        outfile = (STATS_FILE_PATH / results_file_name).open("w")
        outfile.write("{}\n".format(DELIMITER.join(["Checker", "Novel", "Observation", "Consistency"])))
        consistency_checker = consistency_checker_type.value()
        exp_name = DELIMITER.join((consistency_checker_type.name, is_novel))
        run_experiments(exp_name,
                        consistency_checker,
                        obs_files,
                        outfile)
        outfile.close()

    is_novel = "False"
    obs_path = pathlib.Path(observations_path) / 'non_novel'
    obs_files = list(obs_path.glob('*_observation.p'))
    obs_files = random.sample(obs_files, samples)
    for consistency_checker_type in consistency_checker_types:
        results_file_name = results_file_template.format(alg_name=consistency_checker_type.name)
        outfile = (STATS_FILE_PATH / results_file_name).open("a")

        consistency_checker = consistency_checker_type.value()
        checker_name = consistency_checker_type.name
        exp_name = DELIMITER.join((checker_name, is_novel))
        run_experiments(exp_name,
                        consistency_checker,
                        obs_files,
                        outfile)
        outfile.close()

''' Runs the given consistency checker on the set of given set of files and output the results to the results file'''
def run_experiments(exp_name, consistency_checker, obs_files, outfile):
    simulator = CachingPddlPlusSimulator()
    delta_t = settings.SB_DELTA_T
    meta_model = MetaModel()
    for obs_file in obs_files:
        obs = pickle.load(open(obs_file, "rb"))
        consistency_value = check_obs_consistency(obs, meta_model, consistency_checker, simulator=simulator,
                                                  delta_t=delta_t)
        results_line = "{}\n".format(DELIMITER.join((exp_name , obs_file.name, str(consistency_value))))
        logger.info(results_line)
        outfile.write(results_line)
        outfile.flush()

''' Compute the ROC curve of the results file '''
def analyze_results(results_file: pathlib.Path, alg_name = ""):
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
    algs_to_test = [ConsistencyCheckerType.ScienceBirds,
                    ConsistencyCheckerType.BirdLocation,
                    ConsistencyCheckerType.BlockNotDead]
    results_file_template = STATS_FILE_TEMPLATE
    run_consistency_stats(algs_to_test, results_file_template=results_file_template, seed=0)
    logger.info("Experiment Done! Now analyzing...")
    for alg in algs_to_test:
        results_file_name = results_file_template.format(alg_name=alg.name)
        results_file = (STATS_FILE_PATH / results_file_name)

        analyze_results(results_file, alg.name)
    logger.info("Analysis Done!!")