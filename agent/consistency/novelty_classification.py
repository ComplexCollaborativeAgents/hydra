import pandas as pd
import random
import settings
from os import path

def initize_novelty_detector():
    novelty_data = pd.read_csv(path.join(settings.ROOT_PATH, 'data',
                                'science_birds', 'consistency',
                                'novelty.csv'))
    non_novelty_data = pd.read_csv(path.join(settings.ROOT_PATH, 'data',
                                'science_birds', 'consistency',
                                'non_novelty.csv'))

    novelty_unknowns = novelty_data['Unknown'].to_list()
    non_novelty_unknowns = non_novelty_data['Unknown'].to_list()

    ret_dict = {}
    # each row is s(t), s(t-1), s(t-2), Novelty?
    for i in range(100000):
        rand = random.randrange(0,100)
        row = []
        if rand > 80:
            row.append(random.choice(novelty_unknowns))
        else:
            row.append(random.choice(non_novelty_unknowns))
        if rand > 60:
            row.append(random.choice(novelty_unknowns))
        else:
            row.append(random.choice(non_novelty_unknowns))
        if rand > 40:
            row.append(random.choice(novelty_unknowns))
            row.append(1)
        else:
            row.append(random.choice(non_novelty_unknowns))
            row.append(0)
        ret_dict['row_{}'.format(i)] = row


    data = pd.DataFrame.from_dict(ret_dict, orient='index', columns=['s','s-1','s-2','novel'])


    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(data.drop('novel',axis=1),
                                                        data['novel'], test_size=0.10,
                                                        random_state=101)

    from sklearn.linear_model import LogisticRegression
    logmodel = LogisticRegression(solver='liblinear',class_weight='balanced')
    logmodel.fit(X_train,y_train)
    predictions = logmodel.predict(X_test)

    from sklearn.metrics import classification_report
    #print(classification_report(y_test,predictions))
    return logmodel

if __name__ == "__main__":
    model = initize_novelty_detector()