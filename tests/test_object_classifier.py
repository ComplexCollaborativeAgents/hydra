import agent.perception.perception as perception
import csv
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import sklearn.linear_model as lm
import settings
import os

def test_classifier():
    p = perception.Perception()
    with open(os.path.join(settings.ROOT_PATH,'data/science_birds/perception/object_class_level_1.csv'), 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            data = [row[col] for col in perception.classification_cols()]
            type = p.type_to_class(data[0])
            features = [float(i) for i in data[1:]]
            classification = p.classify_obj(features,translate_to_features=False)
            if 'novel' not in type:
                assert type == classification
            elif 'novel' in type:
                assert classification == 'unknown'
            else:
                assert False
    assert True
