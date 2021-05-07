import torch
from ab_fcnn_multichannel import MyCNN
from ab_dataset_tensor import ABDataset
from torch.utils.data import DataLoader
from sklearn.metrics import roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np
import pickle
import pdb
import pandas as pd

device = 'cuda' if torch.cuda.is_available() else 'cpu'
#device = 'cpu'
no_novelty_set = ABDataset('../data_new_preproc', mode='test')
no_novelty_train_size = int(0.8*len(no_novelty_set))
no_novelty_train_set, no_novelty_test_set = torch.utils.data.random_split(no_novelty_set, [no_novelty_train_size, len(no_novelty_set) - no_novelty_train_size])

cnn = MyCNN(device)
cnn.load_state_dict(torch.load('models/pretrained_model_loss.pt', map_location=device))
cnn.eval()

results_df = pd.DataFrame(columns=['Train', 'Test', 'AUC'])

train_test_pairs = [
    # In-level, cross type
    (['Level_1_6', 'Level_1_7'], ['Level_1_8']),
    (['Level_1_6', 'Level_1_8'], ['Level_1_7']),
    (['Level_1_7', 'Level_1_8'], ['Level_1_6']),
    (['Level_2_6', 'Level_2_7', 'Level_2_8', 'Level_2_9'], ['Level_2_10']),
    (['Level_2_6', 'Level_2_7', 'Level_2_8', 'Level_2_10'], ['Level_2_9']),
    (['Level_2_6', 'Level_2_7', 'Level_2_9', 'Level_2_10'], ['Level_2_8']),
    (['Level_2_6', 'Level_2_8', 'Level_2_9', 'Level_2_10'], ['Level_2_7']),
    (['Level_2_7', 'Level_2_8', 'Level_2_9', 'Level_2_10'], ['Level_2_6']),
    (['Level_3_6'], ['Level_3_7']),
    (['Level_3_7'], ['Level_3_6']),
    # All-levels, cross type
    (['Level_1_6', 'Level_1_7', 'Level_2_6', 'Level_2_7', 'Level_2_8', 'Level_2_9', 'Level_2_10', 'Level_3_6', 'Level_3_7'], ['Level_1_8']),
    (['Level_1_6', 'Level_1_8', 'Level_2_6', 'Level_2_7', 'Level_2_8', 'Level_2_9', 'Level_2_10', 'Level_3_6', 'Level_3_7'], ['Level_1_7']),
    (['Level_1_7', 'Level_1_8', 'Level_2_6', 'Level_2_7', 'Level_2_8', 'Level_2_9', 'Level_2_10', 'Level_3_6', 'Level_3_7'], ['Level_1_6']),
    (['Level_1_6', 'Level_1_7', 'Level_1_8', 'Level_2_6', 'Level_2_7', 'Level_2_8', 'Level_2_9', 'Level_3_6', 'Level_3_7'], ['Level_2_10']),
    (['Level_1_6', 'Level_1_7', 'Level_1_8', 'Level_2_6', 'Level_2_7', 'Level_2_8', 'Level_2_10', 'Level_3_6', 'Level_3_7'], ['Level_2_9']),
    (['Level_1_6', 'Level_1_7', 'Level_1_8', 'Level_2_6', 'Level_2_7', 'Level_2_9', 'Level_2_10', 'Level_3_6', 'Level_3_7'], ['Level_2_8']),
    (['Level_1_6', 'Level_1_7', 'Level_1_8', 'Level_2_6', 'Level_2_8', 'Level_2_9', 'Level_2_10', 'Level_3_6', 'Level_3_7'], ['Level_2_7']),
    # (['Level_1_6', 'Level_1_7', 'Level_1_8',' Level_2_7', 'Level_2_8', 'Level_2_9', 'Level_2_10', 'Level_3_6', 'Level_3_7'], ['Level_2_6']),
    (['Level_1_6', 'Level_1_7', 'Level_1_8', 'Level_2_6', 'Level_2_7', 'Level_2_8', 'Level_2_9', 'Level_2_10', 'Level_3_6'], ['Level_3_7']),
    (['Level_1_6', 'Level_1_7', 'Level_1_8', 'Level_2_6', 'Level_2_7', 'Level_2_8', 'Level_2_9', 'Level_2_10', 'Level_3_7'], ['Level_3_6']),
]

no_noveltyloader_train = DataLoader(
    no_novelty_train_set,
    batch_size=512,
    shuffle=True,
    # num_workers=10,
    pin_memory=True,
    # collate_fn=collate_fn
)

no_noveltyloader_test = DataLoader(
    no_novelty_test_set,
    batch_size=512,
    shuffle=True,
    # num_workers=10,
    pin_memory=True,
    # collate_fn=collate_fn
)

with torch.no_grad():
    num_points_changed = 0
    num_points_unchanged = 0
    num_points_predicted_changed = 0
    num_points_predicted_unchanged = 0

    err_changed = 0
    err_unchanged = 0
    err_predicted_changed = 0
    err_predicted_unchanged = 0

    err_changed_no_novelty_train = []
    err_unchanged_no_novelty_train = []
    err_predicted_changed_no_novelty_train = []
    err_predicted_unchanged_no_novelty_train = []

    for states, actions, next_states in no_noveltyloader_train:
        actions = actions.to(device, non_blocking=True)
        states = states.to(device, non_blocking=True)
        next_states = next_states.to(device, non_blocking=True)
        states_resized = states.reshape(states.shape[0], -1)
        next_states_resized = next_states.reshape(next_states.shape[0], -1)
        y_hat = cnn(states_resized, actions)

        mask_changed = torch.ne(states_resized, next_states_resized)   # this will consider only points that change
        mask_predicted_changed = torch.ne(states_resized, y_hat > 0)   # this will consider only points that change

        num_points_changed += mask_changed.sum().item()
        num_points_unchanged += (~mask_changed).sum().item()
        num_points_predicted_changed += mask_predicted_changed.sum().item()
        num_points_predicted_unchanged += (~mask_predicted_changed).sum().item()

        err_changed += (torch.ne(y_hat > 0, next_states_resized) * mask_changed).sum().item()
        err_unchanged += (torch.ne(y_hat > 0, next_states_resized) * (~mask_changed)).sum().item()
        err_predicted_changed += (torch.ne(y_hat > 0, next_states_resized) * mask_predicted_changed).sum().item()
        err_predicted_unchanged += (torch.ne(y_hat > 0, next_states_resized) * (~mask_predicted_changed)).sum().item()

        err_changed_no_novelty_train.append((torch.ne(y_hat > 0, next_states_resized) * mask_changed) )#/ mask_changed.sum(dim=(2,3)))
        err_unchanged_no_novelty_train.append((torch.ne(y_hat > 0, next_states_resized) * (~mask_changed)) )# (~mask_changed).sum(dim=(2,3)))
        err_predicted_changed_no_novelty_train.append((torch.ne(y_hat > 0, next_states_resized) * mask_predicted_changed) )#/ mask_predicted_changed.sum(dim=(2,3)))
        err_predicted_unchanged_no_novelty_train.append((torch.ne(y_hat > 0, next_states_resized) * (~mask_predicted_changed)) ) #/ (~mask_predicted_changed).sum(dim=(2,3)))

    err_changed_no_novelty_train = torch.cat(err_changed_no_novelty_train).data.cpu().numpy()
    err_unchanged_no_novelty_train = torch.cat(err_unchanged_no_novelty_train).data.cpu().numpy()
    err_predicted_changed_no_novelty_train = torch.cat(err_predicted_changed_no_novelty_train).data.cpu().numpy()
    err_predicted_unchanged_no_novelty_train = torch.cat(err_predicted_unchanged_no_novelty_train).data.cpu().numpy()

    print('\t unseen levels, train: err_changed: {:.3f}, err_unchanged: {:.3f}, num_changed: {}'.format(err_changed / num_points_changed,  err_unchanged / num_points_unchanged, num_points_changed))
    print(err_changed_no_novelty_train.shape, err_changed_no_novelty_train.mean(), err_changed_no_novelty_train.std())
    print(err_unchanged_no_novelty_train.shape, err_unchanged_no_novelty_train.mean(), err_unchanged_no_novelty_train.std())
    print(err_predicted_changed_no_novelty_train.shape, err_predicted_changed_no_novelty_train.mean(), err_predicted_changed_no_novelty_train.std())
    print(err_predicted_unchanged_no_novelty_train.shape, err_predicted_unchanged_no_novelty_train.mean(), err_predicted_unchanged_no_novelty_train.std())


    num_points_changed = 0
    num_points_unchanged = 0
    num_points_predicted_changed = 0
    num_points_predicted_unchanged = 0

    err_changed = 0
    err_unchanged = 0
    err_predicted_changed = 0
    err_predicted_unchanged = 0

    err_changed_no_novelty_test = []
    err_unchanged_no_novelty_test = []
    err_predicted_changed_no_novelty_test = []
    err_predicted_unchanged_no_novelty_test = []

    for states, actions, next_states in no_noveltyloader_test:
        actions = actions.to(device, non_blocking=True)
        states = states.to(device, non_blocking=True)
        next_states = next_states.to(device, non_blocking=True)
        states_resized = states.reshape(states.shape[0], -1)
        next_states_resized = next_states.reshape(next_states.shape[0], -1)
        y_hat = cnn(states_resized, actions)

        mask_changed = torch.ne(states_resized, next_states_resized)   # this will consider only points that change
        mask_predicted_changed = torch.ne(states_resized, y_hat > 0)   # this will consider only points that change

        num_points_changed += mask_changed.sum().item()
        num_points_unchanged += (~mask_changed).sum().item()
        num_points_predicted_changed += mask_predicted_changed.sum().item()
        num_points_predicted_unchanged += (~mask_predicted_changed).sum().item()

        err_changed += (torch.ne(y_hat > 0, next_states_resized) * mask_changed).sum().item()
        err_unchanged += (torch.ne(y_hat > 0, next_states_resized) * (~mask_changed)).sum().item()
        err_predicted_changed += (torch.ne(y_hat > 0, next_states_resized) * mask_predicted_changed).sum().item()
        err_predicted_unchanged += (torch.ne(y_hat > 0, next_states_resized) * (~mask_predicted_changed)).sum().item()

        err_changed_no_novelty_test.append((torch.ne(y_hat > 0, next_states_resized) * mask_changed) )#/ mask_changed.sum(dim=(2,3)))
        err_unchanged_no_novelty_test.append((torch.ne(y_hat > 0, next_states_resized) * (~mask_changed)) )# (~mask_changed).sum(dim=(2,3)))
        err_predicted_changed_no_novelty_test.append((torch.ne(y_hat > 0, next_states_resized) * mask_predicted_changed) )#/ mask_predicted_changed.sum(dim=(2,3)))
        err_predicted_unchanged_no_novelty_test.append((torch.ne(y_hat > 0, next_states_resized) * (~mask_predicted_changed)) ) #/ (~mask_predicted_changed).sum(dim=(2,3)))

    err_changed_no_novelty_test = torch.cat(err_changed_no_novelty_test).data.cpu().numpy()
    err_unchanged_no_novelty_test = torch.cat(err_unchanged_no_novelty_test).data.cpu().numpy()
    err_predicted_changed_no_novelty_test = torch.cat(err_predicted_changed_no_novelty_test).data.cpu().numpy()
    err_predicted_unchanged_no_novelty_test = torch.cat(err_predicted_unchanged_no_novelty_test).data.cpu().numpy()

    print('\t unseen levels, test: err_changed: {:.3f}, err_unchanged: {:.3f}, num_changed: {}'.format(err_changed / num_points_changed,  err_unchanged / num_points_unchanged, num_points_changed))
    print(err_changed_no_novelty_test.shape, err_changed_no_novelty_test.mean(), err_changed_no_novelty_test.std())
    print(err_unchanged_no_novelty_test.shape, err_unchanged_no_novelty_test.mean(), err_unchanged_no_novelty_test.std())
    print(err_predicted_changed_no_novelty_test.shape, err_predicted_changed_no_novelty_test.mean(), err_predicted_changed_no_novelty_test.std())
    print(err_predicted_unchanged_no_novelty_train.shape, err_predicted_unchanged_no_novelty_train.mean(), err_predicted_unchanged_no_novelty_train.std())

i = 0
for train, test in train_test_pairs:
    novelty_train_set = torch.utils.data.ConcatDataset([ABDataset('../data_new_preproc', mode=level) for level in train])
    noveltyloader_train = DataLoader(
        novelty_train_set,
        batch_size=512,
        shuffle=True,
        # num_workers=10,
        pin_memory=True,
        # collate_fn=collate_fn
    )

    novelty_test_set = torch.utils.data.ConcatDataset([ABDataset('../data_new_preproc', mode=level) for level in test])
    noveltyloader_test = DataLoader(
        novelty_test_set,
        batch_size=512,
        shuffle=True,
        # num_workers=10,
        pin_memory=True,
        # collate_fn=collate_fn
    )

    with torch.no_grad():
        num_points_changed = 0
        num_points_unchanged = 0
        num_points_predicted_changed = 0
        num_points_predicted_unchanged = 0

        err_changed = 0
        err_unchanged = 0
        err_predicted_changed = 0
        err_predicted_unchanged = 0

        err_changed_novelty_train = []
        err_unchanged_novelty_train = []
        err_predicted_changed_novelty_train = []
        err_predicted_unchanged_novelty_train = []

        for states, actions, next_states in noveltyloader_train:
            actions = actions.to(device, non_blocking=True)
            states = states.to(device, non_blocking=True)
            next_states = next_states.to(device, non_blocking=True)
            states_resized = states.reshape(states.shape[0], -1)
            next_states_resized = next_states.reshape(next_states.shape[0], -1)
            y_hat = cnn(states_resized, actions)

            mask_changed = torch.ne(states_resized, next_states_resized)   # this will consider only points that change
            mask_predicted_changed = torch.ne(states_resized, y_hat > 0)   # this will consider only points that change

            num_points_changed += mask_changed.sum().item()
            num_points_unchanged += (~mask_changed).sum().item()
            num_points_predicted_changed += mask_predicted_changed.sum().item()
            num_points_predicted_unchanged += (~mask_predicted_changed).sum().item()

            err_changed += (torch.ne(y_hat > 0, next_states_resized) * mask_changed).sum().item()
            err_unchanged += (torch.ne(y_hat > 0, next_states_resized) * (~mask_changed)).sum().item()
            err_predicted_changed += (torch.ne(y_hat > 0, next_states_resized) * mask_predicted_changed).sum().item()
            err_predicted_unchanged += (torch.ne(y_hat > 0, next_states_resized) * (~mask_predicted_changed)).sum().item()

            err_changed_novelty_train.append((torch.ne(y_hat > 0, next_states_resized) * mask_changed) )#/ mask_changed.sum(dim=(2,3)))
            err_unchanged_novelty_train.append((torch.ne(y_hat > 0, next_states_resized) * (~mask_changed)) )# (~mask_changed).sum(dim=(2,3)))
            err_predicted_changed_novelty_train.append((torch.ne(y_hat > 0, next_states_resized) * mask_predicted_changed) )#/ mask_predicted_changed.sum(dim=(2,3)))
            err_predicted_unchanged_novelty_train.append((torch.ne(y_hat > 0, next_states_resized) * (~mask_predicted_changed)) ) #/ (~mask_predicted_changed).sum(dim=(2,3)))

        err_changed_novelty_train = torch.cat(err_changed_novelty_train).data.cpu().numpy()
        err_unchanged_novelty_train = torch.cat(err_unchanged_novelty_train).data.cpu().numpy()
        err_predicted_changed_novelty_train = torch.cat(err_predicted_changed_novelty_train).data.cpu().numpy()
        err_predicted_unchanged_novelty_train = torch.cat(err_predicted_unchanged_novelty_train).data.cpu().numpy()

        print('\t novelty levels: err_changed: {:.3f}, err_unchanged: {:.3f}, num_changed: {}'.format(err_changed / num_points_changed,  err_unchanged / num_points_unchanged, num_points_changed))
        print(err_changed_novelty_train.shape, err_changed_novelty_train.mean(), err_changed_novelty_train.std())
        print(err_unchanged_novelty_train.shape, err_unchanged_novelty_train.mean(), err_unchanged_novelty_train.std())
        print(err_predicted_changed_novelty_train.shape, err_predicted_changed_novelty_train.mean(), err_predicted_changed_novelty_train.std())
        print(err_predicted_unchanged_novelty_train.shape, err_predicted_unchanged_novelty_train.mean(), err_predicted_unchanged_novelty_train.std())

        num_points_changed = 0
        num_points_unchanged = 0
        num_points_predicted_changed = 0
        num_points_predicted_unchanged = 0

        err_changed = 0
        err_unchanged = 0
        err_predicted_changed = 0
        err_predicted_unchanged = 0

        err_changed_novelty_test = []
        err_unchanged_novelty_test = []
        err_predicted_changed_novelty_test = []
        err_predicted_unchanged_novelty_test = []

        for states, actions, next_states in noveltyloader_test:
            actions = actions.to(device, non_blocking=True)
            states = states.to(device, non_blocking=True)
            next_states = next_states.to(device, non_blocking=True)
            states_resized = states.reshape(states.shape[0], -1)
            next_states_resized = next_states.reshape(next_states.shape[0], -1)
            y_hat = cnn(states_resized, actions)

            mask_changed = torch.ne(states_resized, next_states_resized)   # this will consider only points that change
            mask_predicted_changed = torch.ne(states_resized, y_hat > 0)   # this will consider only points that change

            num_points_changed += mask_changed.sum().item()
            num_points_unchanged += (~mask_changed).sum().item()
            num_points_predicted_changed += mask_predicted_changed.sum().item()
            num_points_predicted_unchanged += (~mask_predicted_changed).sum().item()

            err_changed += (torch.ne(y_hat > 0, next_states_resized) * mask_changed).sum().item()
            err_unchanged += (torch.ne(y_hat > 0, next_states_resized) * (~mask_changed)).sum().item()
            err_predicted_changed += (torch.ne(y_hat > 0, next_states_resized) * mask_predicted_changed).sum().item()
            err_predicted_unchanged += (torch.ne(y_hat > 0, next_states_resized) * (~mask_predicted_changed)).sum().item()

            err_changed_novelty_test.append((torch.ne(y_hat > 0, next_states_resized) * mask_changed) )#/ mask_changed.sum(dim=(2,3)))
            err_unchanged_novelty_test.append((torch.ne(y_hat > 0, next_states_resized) * (~mask_changed)) )# (~mask_changed).sum(dim=(2,3)))
            err_predicted_changed_novelty_test.append((torch.ne(y_hat > 0, next_states_resized) * mask_predicted_changed) )#/ mask_predicted_changed.sum(dim=(2,3)))
            err_predicted_unchanged_novelty_test.append((torch.ne(y_hat > 0, next_states_resized) * (~mask_predicted_changed)) ) #/ (~mask_predicted_changed).sum(dim=(2,3)))

        err_changed_novelty_test = torch.cat(err_changed_novelty_test).data.cpu().numpy()
        err_unchanged_novelty_test = torch.cat(err_unchanged_novelty_test).data.cpu().numpy()
        err_predicted_changed_novelty_test = torch.cat(err_predicted_changed_novelty_test).data.cpu().numpy()
        err_predicted_unchanged_novelty_test = torch.cat(err_predicted_unchanged_novelty_test).data.cpu().numpy()

        print('\t novelty levels: err_changed: {:.3f}, err_unchanged: {:.3f}, num_changed: {}'.format(err_changed / num_points_changed,  err_unchanged / num_points_unchanged, num_points_changed))
        print(err_changed_novelty_test.shape, err_changed_novelty_test.mean(), err_changed_novelty_test.std())
        print(err_unchanged_novelty_test.shape, err_unchanged_novelty_test.mean(), err_unchanged_novelty_test.std())
        print(err_predicted_changed_novelty_test.shape, err_predicted_changed_novelty_test.mean(), err_predicted_changed_novelty_test.std())
        print(err_predicted_unchanged_novelty_test.shape, err_predicted_unchanged_novelty_test.mean(), err_predicted_unchanged_novelty_test.std())


    y_true_train = np.r_[np.zeros(len(no_novelty_train_set)), np.ones(len(novelty_train_set))]
    X_novelty_train = np.c_[np.r_[err_changed_no_novelty_train, err_changed_novelty_train],
                        np.r_[err_unchanged_no_novelty_train, err_unchanged_novelty_train],
                        np.r_[err_predicted_changed_no_novelty_train, err_predicted_changed_novelty_train],
                        np.r_[err_predicted_unchanged_no_novelty_train, err_predicted_unchanged_novelty_train]]
    X_novelty_train[np.isnan(X_novelty_train)] = 0
    X_mean = X_novelty_train.mean(axis=0)
    X_std = X_novelty_train.std(axis=0)
    X_std[X_std == 0] = 1
    X_novelty_train = (X_novelty_train - X_mean) / X_std

    y_true_test = np.r_[np.zeros(len(no_novelty_test_set)), np.ones(len(novelty_test_set))]
    X_novelty_test = np.c_[np.r_[err_changed_no_novelty_test, err_changed_novelty_test],
                        np.r_[err_unchanged_no_novelty_test, err_unchanged_novelty_test],
                        np.r_[err_predicted_changed_no_novelty_test, err_predicted_changed_novelty_test],
                        np.r_[err_predicted_unchanged_no_novelty_test, err_predicted_unchanged_novelty_test]]
    X_novelty_test[np.isnan(X_novelty_test)] = 0
    X_novelty_test = (X_novelty_test - X_mean) / X_std


    print(X_novelty_train.max(axis=0), X_novelty_train.min(axis=0))
    print(X_novelty_train.shape)

    clf = LogisticRegression(penalty='none', class_weight='balanced', max_iter=1000)
    clf.fit(X_novelty_train, y_true_train)
    with open('fc_pretrained_novelty_detector_{}.pickle'.format(i), 'wb') as f:
        pickle.dump({'clf': clf, 'x_mu': X_mean, 'x_std': X_std}, f)

    y_score = clf.predict_proba(X_novelty_test)[:, 1]

    # y_score = np.r_[err_changed_no_novelty_train.data.cpu().numpy(), err_changed_novelty_train.data.cpu().numpy()]
    # y_score = np.r_[err_predicted_changed_no_novelty_train.data.cpu().numpy(), err_predicted_changed_novelty_train.data.cpu().numpy()]
    print(y_true_test.shape, y_score.shape)

    fpr, tpr, thresholds = roc_curve(y_true_test, y_score)
    print(fpr.shape, tpr.shape)

    plt.figure()
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], color='k', linestyle='--')
    plt.ylim(0, 1)
    plt.xlim(0, 1)
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('Train: {}, Test: {}'.format(train, test), wrap=True)
    plt.savefig('fc_roc_curve_novelty_{}.pdf'.format(i))

    auc = metrics.roc_auc_score(y_true_test, y_score)

    print('{}-th model:'.format(i))
    print('\tTrain: {}'.format(train))
    print('\tTest: {}'.format(test))
    print('\tAUC: {}'.format(auc))

    results_df = results_df.append({'Train': train, 'Test': test, 'AUC': auc}, ignore_index=True)

    i += 1

#print(results_df.to_markdown())

################# FINAL MODEL ################

no_noveltyloader = DataLoader(
    no_novelty_set,
    batch_size=32,
    shuffle=True,
    # num_workers=10,
    pin_memory=True,
    # collate_fn=collate_fn
)
with torch.no_grad():
    num_points_changed = 0
    num_points_unchanged = 0
    num_points_predicted_changed = 0
    num_points_predicted_unchanged = 0

    err_changed = 0
    err_unchanged = 0
    err_predicted_changed = 0
    err_predicted_unchanged = 0

    err_changed_no_novelty_train = []
    err_unchanged_no_novelty_train = []
    err_predicted_changed_no_novelty_train = []
    err_predicted_unchanged_no_novelty_train = []

    for states, actions, next_states in no_noveltyloader:
        actions = actions.to(device, non_blocking=True)
        states = states.to(device, non_blocking=True)
        next_states = next_states.to(device, non_blocking=True)
        states_resized = states.reshape(states.shape[0], -1)
        next_states_resized = next_states.reshape(next_states.shape[0], -1)
        y_hat = cnn(states_resized, actions)

        mask_changed = torch.ne(states_resized, next_states_resized)   # this will consider only points that change
        mask_predicted_changed = torch.ne(states_resized, y_hat > 0)   # this will consider only points that change

        num_points_changed += mask_changed.sum().item()
        num_points_unchanged += (~mask_changed).sum().item()
        num_points_predicted_changed += mask_predicted_changed.sum().item()
        num_points_predicted_unchanged += (~mask_predicted_changed).sum().item()

        err_changed += (torch.ne(y_hat > 0, next_states_resized) * mask_changed).sum().item()
        err_unchanged += (torch.ne(y_hat > 0, next_states_resized) * (~mask_changed)).sum().item()
        err_predicted_changed += (torch.ne(y_hat > 0, next_states_resized) * mask_predicted_changed).sum().item()
        err_predicted_unchanged += (torch.ne(y_hat > 0, next_states_resized) * (~mask_predicted_changed)).sum().item()

        err_changed_no_novelty_train.append((torch.ne(y_hat > 0, next_states_resized) * mask_changed) )#/ mask_changed.sum(dim=(2,3)))
        err_unchanged_no_novelty_train.append((torch.ne(y_hat > 0, next_states_resized) * (~mask_changed)) )# (~mask_changed).sum(dim=(2,3)))
        err_predicted_changed_no_novelty_train.append((torch.ne(y_hat > 0, next_states_resized) * mask_predicted_changed) )#/ mask_predicted_changed.sum(dim=(2,3)))
        err_predicted_unchanged_no_novelty_train.append((torch.ne(y_hat > 0, next_states_resized) * (~mask_predicted_changed)) ) #/ (~mask_predicted_changed).sum(dim=(2,3)))

    err_changed_no_novelty_train = torch.cat(err_changed_no_novelty_train).data.cpu().numpy()
    err_unchanged_no_novelty_train = torch.cat(err_unchanged_no_novelty_train).data.cpu().numpy()
    err_predicted_changed_no_novelty_train = torch.cat(err_predicted_changed_no_novelty_train).data.cpu().numpy()
    err_predicted_unchanged_no_novelty_train = torch.cat(err_predicted_unchanged_no_novelty_train).data.cpu().numpy()

    print('\t unseen levels: err_changed: {:.3f}, err_unchanged: {:.3f}, num_changed: {}'.format(err_changed / num_points_changed,  err_unchanged / num_points_unchanged, num_points_changed))
    print(err_changed_no_novelty_train.shape, err_changed_no_novelty_train.mean(), err_changed_no_novelty_train.std())
    print(err_unchanged_no_novelty_train.shape, err_unchanged_no_novelty_train.mean(), err_unchanged_no_novelty_train.std())
    print(err_predicted_changed_no_novelty_train.shape, err_predicted_changed_no_novelty_train.mean(), err_predicted_changed_no_novelty_train.std())
    print(err_predicted_unchanged_no_novelty_train.shape, err_predicted_unchanged_no_novelty_train.mean(), err_predicted_unchanged_no_novelty_train.std())


all_novelties = ['Level_1_6', 'Level_1_7', 'Level_1_8', 'Level_2_6', 'Level_2_7', 'Level_2_8', 'Level_2_9', 'Level_2_10', 'Level_3_6', 'Level_3_7']
novelty_set = torch.utils.data.ConcatDataset([ABDataset('../data_new_preproc', mode=level) for level in all_novelties])
noveltyloader = DataLoader(
    novelty_set,
    batch_size=2,
    shuffle=True,
    # num_workers=10,
    pin_memory=True,
    # collate_fn=collate_fn
)

with torch.no_grad():
    num_points_changed = 0
    num_points_unchanged = 0
    num_points_predicted_changed = 0
    num_points_predicted_unchanged = 0

    err_changed = 0
    err_unchanged = 0
    err_predicted_changed = 0
    err_predicted_unchanged = 0

    err_changed_novelty_train = []
    err_unchanged_novelty_train = []
    err_predicted_changed_novelty_train = []
    err_predicted_unchanged_novelty_train = []

    for states, actions, next_states in noveltyloader:
        actions = actions.to(device, non_blocking=True)
        states = states.to(device, non_blocking=True)
        next_states = next_states.to(device, non_blocking=True)
        states_resized = states.reshape(states.shape[0], -1)
        next_states_resized = next_states.reshape(next_states.shape[0], -1)
        y_hat = cnn(states_resized, actions)

        mask_changed = torch.ne(states_resized, next_states_resized)   # this will consider only points that change
        mask_predicted_changed = torch.ne(states_resized, y_hat > 0)   # this will consider only points that change

        num_points_changed += mask_changed.sum().item()
        num_points_unchanged += (~mask_changed).sum().item()
        num_points_predicted_changed += mask_predicted_changed.sum().item()
        num_points_predicted_unchanged += (~mask_predicted_changed).sum().item()

        err_changed += (torch.ne(y_hat > 0, next_states_resized) * mask_changed).sum().item()
        err_unchanged += (torch.ne(y_hat > 0, next_states_resized) * (~mask_changed)).sum().item()
        err_predicted_changed += (torch.ne(y_hat > 0, next_states_resized) * mask_predicted_changed).sum().item()
        err_predicted_unchanged += (torch.ne(y_hat > 0, next_states_resized) * (~mask_predicted_changed)).sum().item()

        err_changed_novelty_train.append((torch.ne(y_hat > 0, next_states_resized) * mask_changed) )#/ mask_changed.sum(dim=(2,3)))
        err_unchanged_novelty_train.append((torch.ne(y_hat > 0, next_states_resized) * (~mask_changed)) )# (~mask_changed).sum(dim=(2,3)))
        err_predicted_changed_novelty_train.append((torch.ne(y_hat > 0, next_states_resized) * mask_predicted_changed) )#/ mask_predicted_changed.sum(dim=(2,3)))
        err_predicted_unchanged_novelty_train.append((torch.ne(y_hat > 0, next_states_resized) * (~mask_predicted_changed)) ) #/ (~mask_predicted_changed).sum(dim=(2,3)))

    err_changed_novelty_train = torch.cat(err_changed_novelty_train).data.cpu().numpy()
    err_unchanged_novelty_train = torch.cat(err_unchanged_novelty_train).data.cpu().numpy()
    err_predicted_changed_novelty_train = torch.cat(err_predicted_changed_novelty_train).data.cpu().numpy()
    err_predicted_unchanged_novelty_train = torch.cat(err_predicted_unchanged_novelty_train).data.cpu().numpy()

    print('\t novelty levels: err_changed: {:.3f}, err_unchanged: {:.3f}, num_changed: {}'.format(err_changed / num_points_changed,  err_unchanged / num_points_unchanged, num_points_changed))
    print(err_changed_novelty_train.shape, err_changed_novelty_train.mean(), err_changed_novelty_train.std())
    print(err_unchanged_novelty_train.shape, err_unchanged_novelty_train.mean(), err_unchanged_novelty_train.std())
    print(err_predicted_changed_novelty_train.shape, err_predicted_changed_novelty_train.mean(), err_predicted_changed_novelty_train.std())
    print(err_predicted_unchanged_novelty_train.shape, err_predicted_unchanged_novelty_train.mean(), err_predicted_unchanged_novelty_train.std())


y_true = np.r_[np.zeros(len(no_novelty_set)), np.ones(len(novelty_set))]
X_novelty = np.c_[np.r_[err_changed_no_novelty_train, err_changed_novelty_train],
                    np.r_[err_unchanged_no_novelty_train, err_unchanged_novelty_train],
                    np.r_[err_predicted_changed_no_novelty_train, err_predicted_changed_novelty_train],
                    np.r_[err_predicted_unchanged_no_novelty_train, err_predicted_unchanged_novelty_train]]
X_novelty[np.isnan(X_novelty)] = 0
X_mean = X_novelty.mean(axis=0)
X_std = X_novelty.std(axis=0)
X_std[X_std == 0] = 1
X_novelty = (X_novelty - X_mean) / X_std

clf = LogisticRegression(penalty='none', class_weight='balanced', max_iter=1000)
clf.fit(X_novelty, y_true)
with open('fc_pretrained_novelty_detector.pickle', 'wb') as f:
    pickle.dump({'clf': clf, 'x_mu': X_mean, 'x_std': X_std}, f)
