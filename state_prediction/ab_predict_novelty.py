import torch
from ab_cnn import MyCNN
from ab_dataset_tensor import ABDataset
from torch.utils.data import DataLoader
from sklearn.metrics import roc_curve
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'

test_set = ABDataset('data_test_train_preproc', mode='test')
novelty_set = ABDataset('data_test_train_preproc', mode='novelty_new')

testloader = DataLoader(
    test_set,
    batch_size=512,
    shuffle=True,
    # num_workers=10,
    pin_memory=True,
    # collate_fn=collate_fn
)  
noveltyloader = DataLoader(
    novelty_set,
    batch_size=512,
    shuffle=True,
    # num_workers=10,
    pin_memory=True,
    # collate_fn=collate_fn
)  

cnn = MyCNN(device)
cnn.load_state_dict(torch.load('pretrained_model.pt', map_location=device))
cnn.eval()

with torch.no_grad():
    num_points_changed = 0
    num_points_unchanged = 0
    num_points_predicted_changed = 0
    num_points_predicted_unchanged = 0

    err_changed = 0
    err_unchanged = 0
    err_predicted_changed = 0
    err_predicted_unchanged = 0

    err_changed_no_novelty = []
    err_unchanged_no_novelty = []
    err_predicted_changed_no_novelty = []
    err_predicted_unchanged_no_novelty = []

    for states, actions, next_states in testloader:
        actions = actions.to(device, non_blocking=True)
        states = states.to(device, non_blocking=True)
        next_states = next_states.to(device, non_blocking=True)
        y_hat = cnn(states, actions)
        
        mask_changed = torch.ne(states, next_states)   # this will consider only points that change
        mask_predicted_changed = torch.ne(states, y_hat > 0)   # this will consider only points that change
        
        num_points_changed += mask_changed.sum().item()
        num_points_unchanged += (~mask_changed).sum().item()
        num_points_predicted_changed += mask_predicted_changed.sum().item()
        num_points_predicted_unchanged += (~mask_predicted_changed).sum().item()
    
        err_changed += (torch.ne(y_hat > 0, next_states) * mask_changed).sum().item()
        err_unchanged += (torch.ne(y_hat > 0, next_states) * (~mask_changed)).sum().item()
        err_predicted_changed += (torch.ne(y_hat > 0, next_states) * mask_predicted_changed).sum().item()
        err_predicted_unchanged += (torch.ne(y_hat > 0, next_states) * (~mask_predicted_changed)).sum().item()

        err_changed_no_novelty.append((torch.ne(y_hat > 0, next_states) * mask_changed).sum(dim=(2,3)) / mask_changed.sum(dim=(2,3)))
        err_unchanged_no_novelty.append((torch.ne(y_hat > 0, next_states) * (~mask_changed)).sum(dim=(2,3)) / (~mask_changed).sum(dim=(2,3)))
        err_predicted_changed_no_novelty.append((torch.ne(y_hat > 0, next_states) * mask_predicted_changed).sum(dim=(2,3)) / mask_predicted_changed.sum(dim=(2,3)))
        err_predicted_unchanged_no_novelty.append((torch.ne(y_hat > 0, next_states) * (~mask_predicted_changed)).sum(dim=(2,3)) / (~mask_predicted_changed).sum(dim=(2,3)))

    err_changed_no_novelty = torch.cat(err_changed_no_novelty).data.cpu().numpy()
    err_unchanged_no_novelty = torch.cat(err_unchanged_no_novelty).data.cpu().numpy()
    err_predicted_changed_no_novelty = torch.cat(err_predicted_changed_no_novelty).data.cpu().numpy()
    err_predicted_unchanged_no_novelty = torch.cat(err_predicted_unchanged_no_novelty).data.cpu().numpy()

    print('\t unseen levels: err_changed: {:.3f}, err_unchanged: {:.3f}, num_changed: {}'.format(err_changed / num_points_changed,  err_unchanged / num_points_unchanged, num_points_changed))
    print(err_changed_no_novelty.shape, err_changed_no_novelty.mean(), err_changed_no_novelty.std())
    print(err_unchanged_no_novelty.shape, err_unchanged_no_novelty.mean(), err_unchanged_no_novelty.std())
    print(err_predicted_changed_no_novelty.shape, err_predicted_changed_no_novelty.mean(), err_predicted_changed_no_novelty.std())
    print(err_predicted_unchanged_no_novelty.shape, err_predicted_unchanged_no_novelty.mean(), err_predicted_unchanged_no_novelty.std())

    num_points_changed = 0
    num_points_unchanged = 0
    num_points_predicted_changed = 0
    num_points_predicted_unchanged = 0

    err_changed = 0
    err_unchanged = 0
    err_predicted_changed = 0
    err_predicted_unchanged = 0

    err_changed_novelty = []
    err_unchanged_novelty = []
    err_predicted_changed_novelty = []
    err_predicted_unchanged_novelty = []

    for states, actions, next_states in noveltyloader:
        actions = actions.to(device, non_blocking=True)
        states = states.to(device, non_blocking=True)
        next_states = next_states.to(device, non_blocking=True)
        y_hat = cnn(states, actions)

        mask_changed = torch.ne(states, next_states)   # this will consider only points that change
        mask_predicted_changed = torch.ne(states, y_hat > 0)   # this will consider only points that change
        
        num_points_changed += mask_changed.sum().item()
        num_points_unchanged += (~mask_changed).sum().item()
        num_points_predicted_changed += mask_predicted_changed.sum().item()
        num_points_predicted_unchanged += (~mask_predicted_changed).sum().item()
    
        err_changed += (torch.ne(y_hat > 0, next_states) * mask_changed).sum().item()
        err_unchanged += (torch.ne(y_hat > 0, next_states) * (~mask_changed)).sum().item()
        err_predicted_changed += (torch.ne(y_hat > 0, next_states) * mask_predicted_changed).sum().item()
        err_predicted_unchanged += (torch.ne(y_hat > 0, next_states) * (~mask_predicted_changed)).sum().item()

        err_changed_novelty.append((torch.ne(y_hat > 0, next_states) * mask_changed).sum(dim=(2,3)) / mask_changed.sum(dim=(2,3)))
        err_unchanged_novelty.append((torch.ne(y_hat > 0, next_states) * (~mask_changed)).sum(dim=(2,3)) / (~mask_changed).sum(dim=(2,3)))
        err_predicted_changed_novelty.append((torch.ne(y_hat > 0, next_states) * mask_predicted_changed).sum(dim=(2,3)) / mask_predicted_changed.sum(dim=(2,3)))
        err_predicted_unchanged_novelty.append((torch.ne(y_hat > 0, next_states) * (~mask_predicted_changed)).sum(dim=(2,3)) / (~mask_predicted_changed).sum(dim=(2,3)))

    err_changed_novelty = torch.cat(err_changed_novelty).data.cpu().numpy()
    err_unchanged_novelty = torch.cat(err_unchanged_novelty).data.cpu().numpy()
    err_predicted_changed_novelty = torch.cat(err_predicted_changed_novelty).data.cpu().numpy()
    err_predicted_unchanged_novelty = torch.cat(err_predicted_unchanged_novelty).data.cpu().numpy()
    
    print('\t novelty levels: err_changed: {:.3f}, err_unchanged: {:.3f}, num_changed: {}'.format(err_changed / num_points_changed,  err_unchanged / num_points_unchanged, num_points_changed))
    print(err_changed_novelty.shape, err_changed_novelty.mean(), err_changed_novelty.std())
    print(err_unchanged_novelty.shape, err_unchanged_novelty.mean(), err_unchanged_novelty.std())
    print(err_predicted_changed_novelty.shape, err_predicted_changed_novelty.mean(), err_predicted_changed_novelty.std())
    print(err_predicted_unchanged_novelty.shape, err_predicted_unchanged_novelty.mean(), err_predicted_unchanged_novelty.std())


y_true = np.r_[np.zeros(len(test_set)), np.ones(len(novelty_set))]
X_novelty = np.c_[np.r_[err_changed_no_novelty, err_changed_novelty],
                    np.r_[err_unchanged_no_novelty, err_unchanged_novelty],
                    np.r_[err_predicted_changed_no_novelty, err_predicted_changed_novelty],
                    np.r_[err_predicted_unchanged_no_novelty, err_predicted_unchanged_novelty]]
X_novelty[np.isnan(X_novelty)] = 0
X_novelty = X_novelty - X_novelty.mean(axis=0)
X_novelty = X_novelty / X_novelty.std(axis=0)


print(X_novelty.max(axis=0), X_novelty.min(axis=0))
print(X_novelty.shape)

shuffle_idx = np.random.permutation(y_true.shape[0])
idx_train = shuffle_idx[:int(0.8 * y_true.shape[0])]
idx_test = shuffle_idx[int(0.8 * y_true.shape[0]):]
y_true_train = y_true[idx_train]
X_novelty_train = X_novelty[idx_train]

y_true_test = y_true[idx_test]
X_novelty_test = X_novelty[idx_test]

clf = LogisticRegression(penalty='none', class_weight='balanced', max_iter=1000)
clf.fit(X_novelty_train, y_true_train)
y_score = clf.predict_proba(X_novelty_test)[:, 1]

# y_score = np.r_[err_changed_no_novelty.data.cpu().numpy(), err_changed_novelty.data.cpu().numpy()]
# y_score = np.r_[err_predicted_changed_no_novelty.data.cpu().numpy(), err_predicted_changed_novelty.data.cpu().numpy()]
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
plt.savefig('roc_curve_novelty.pdf')



