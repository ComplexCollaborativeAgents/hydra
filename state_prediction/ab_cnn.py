import torch
import torch.nn as nn

class MyCNN(nn.Module):
    def __init__(self, device='cpu'):
        super().__init__()

        self.conv0 = nn.Sequential(
            nn.Conv2d(13, 16, kernel_size=3),
            nn.MaxPool2d(2),
            # nn.BatchNorm2d(16),
            nn.ReLU(),
            # nn.Dropout(0.5)
        )

        self.conv1 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=(1,2)),
            nn.MaxPool2d(2),
            # nn.BatchNorm2d(32),
            nn.ReLU(),
            # nn.Dropout(0.5)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3),
            nn.MaxPool2d(2),
            # nn.BatchNorm2d(64),
            nn.ReLU(),
            # nn.Dropout(0.5)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3),
            nn.MaxPool2d(2),
            # nn.BatchNorm2d(128),
            nn.ReLU(),
            # nn.Dropout(0.5)
        )

        self.fc0 = nn.Sequential(
            nn.Linear(128, 2048),
            nn.ReLU()
        )

        # self.w_dec = nn.Parameter(torch.randn(2048, 2048))
        # self.w_enc = nn.Parameter(torch.randn(2048, 2048))
        # self.w_u = nn.Parameter(torch.randn(2048, 5))
        # self.b = nn.Parameter(torch.randn(2048))
        self.dec = nn.Linear(2048, 2048, bias=True)
        self.x_enc = nn.Linear(2048, 2048, bias=False)
        self.u_enc = nn.Linear(5, 2048, bias=False)

        self.fc1 = nn.Sequential(
            nn.Linear(2048, 128),
            nn.ReLU()
        )

        self.deconv0 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, output_padding=1),
            # nn.BatchNorm2d(64),
            nn.ReLU(),
            # nn.Dropout(0.5)
        )

        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, output_padding=1),
            # nn.BatchNorm2d(32),
            nn.ReLU(),
            # nn.Dropout(0.5)
        )

        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=(2,4), dilation=2),
            # nn.BatchNorm2d(16),
            nn.ReLU(),
            # nn.Dropout(0.5)
        )

        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(16, 13, kernel_size=3, stride=2, output_padding=1),
            nn.ReLU(),
            # nn.BatchNorm2d(13),
        )

        self.to(device)


    def forward(self, x, u):
        # print('----Original----')
        # print(x.shape)
        # print('----Conv----')
        # x = torch.cat([self.preconv(x), self.preact(u).view(-1, 1, x.shape[2], x.shape[3])], dim=1)

        x = self.conv0(x)
        # print(x.shape)
        x = self.conv1(x)
        # print(x.shape)
        x = self.conv2(x)
        # print(x.shape)
        x = self.conv3(x)
        # print(x.shape)

        # print('----Concat Action----')
        # x = torch.cat((x, u.view(*u.shape, 1, 1)), dim=1)
        x = self.fc0(x.squeeze())
        # x = (x.matmul(self.w_enc.t()) * u.matmul(self.w_u.t())).matmul(self.w_dec.t()) + self.b   # https://arxiv.org/pdf/1507.08750.pdf Sec 3.2
        x = self.dec(self.x_enc(x) * self.u_enc(u))   # https://arxiv.org/pdf/1507.08750.pdf Sec 3.2
        x = self.fc1(x)
        x = x.view(*x.shape, 1, 1)
        # print(x.shape)

        # print(x.shape)
        # x = self.fc0(x.view(x.shape[0], -1)).view(*x.shape)

        # print('----Deconv----')
        x = self.deconv0(x)
        # print(x.shape)
        x = self.deconv1(x)
        # print(x.shape)
        x = self.deconv2(x)
        # print(x.shape)
        x = self.deconv3(x)
        # print(x.shape)
        # exit()

        return x


if __name__ == '__main__':
    import torchvision.transforms as transforms
    from torch.utils.data import DataLoader, random_split
    from ab_dataset_tensor import ABDataset
    # from ab_dataset import ABDataset, collate_fn
    import numpy as np

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # transform = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Resize((48, 84)),
    #     transforms.Lambda(lambda x: (x > 0.5).float())
    # ]
    # )
    # transform = transforms.Lambda(lambda x: torch.from_numpy(x.transpose(2, 0, 1)))
    trainval_set = ABDataset('data_test_train_preproc', mode='train')
    test_set = ABDataset('data_test_train_preproc', mode='test')
    novelty_set = ABDataset('data_test_train_preproc', mode='novelty')
    # torch_dataset = ABDataset('obs_data_preproc', transform)
    n_train = int(0.8 * len(trainval_set))
    n_val = len(trainval_set) - n_train
    train_set, val_set = random_split(trainval_set, [n_train, n_val])
    
    print(len(train_set), len(val_set), len(test_set), len(novelty_set))

    trainloader = DataLoader(
        train_set,
        batch_size=32,
        shuffle=True,
        # num_workers=10,
        pin_memory=True,
        # collate_fn=collate_fn
    )
    valloader = DataLoader(
        val_set,
        batch_size=512,
        shuffle=True,
        # num_workers=10,
        pin_memory=True,
        # collate_fn=collate_fn
    )  
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
    cnn.train()

    loss_fn = nn.BCEWithLogitsLoss(reduction='none')        # we'll need the full tensor loss to be able to apply a mask to it
    optimizer = torch.optim.Adam(cnn.parameters())


    for i in range(1000):
        for states, actions, next_states in trainloader:
        # for obs in trainloader:
            # states, actions, next_states = obs['state'], obs['action'], obs['next_states']
            # print(torch.ne(states, next_states).float().mean(), torch.ne(states, next_states).float().sum(), states.shape)
            # exit()
            actions = actions.to(device, non_blocking=True)
            states = states.to(device, non_blocking=True)
            next_states = next_states.to(device, non_blocking=True)
            y_hat = cnn(states, actions)

            mask_changed = torch.ne(states, next_states)   # this will consider only points that change
            num_points_changed = mask_changed.sum().item()
            num_points_unchanged = (~mask_changed).sum().item()
            loss_raw = loss_fn(y_hat, next_states) 
            loss_changed = (loss_raw * mask_changed).sum() / num_points_changed
            loss_unchanged = (loss_raw * (~mask_changed)).sum() / num_points_unchanged
            loss = loss_changed + 1e-2*loss_unchanged
            # print(loss_changed.item(), loss_unchanged.item())
            # print((torch.ne(y_hat > 0, next_states) * mask_changed).sum().item() / num_points_changed, (torch.ne(y_hat > 0, next_states) * (~mask_changed)).sum().item() / num_points_unchanged, num_points_changed)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            cnn.eval()
            print('Epoch: {}'.format(i))

            num_points_changed = 0
            num_points_unchanged = 0
            err_changed = 0
            err_unchanged = 0
            for states, actions, next_states in valloader:
            # for obs in testloader:
                # states, actions, next_states = obs['state'], obs['action'], obs['next_states']
                actions = actions.to(device, non_blocking=True)
                states = states.to(device, non_blocking=True)
                next_states = next_states.to(device, non_blocking=True)
                y_hat = cnn(states, actions)
                mask_changed = torch.ne(states, next_states)   # this will consider only points that change
                num_points_changed += mask_changed.sum().item()
                num_points_unchanged += (~mask_changed).sum().item()
            
                err_changed += (torch.ne(y_hat > 0, next_states) * mask_changed).sum().item()
                err_unchanged += (torch.ne(y_hat > 0, next_states) * (~mask_changed)).sum().item()

            print('\t same levels: err_changed: {}, err_unchanged: {}, num_changed: {}'.format(err_changed / num_points_changed,  err_unchanged / num_points_unchanged, num_points_changed))
            
            num_points_changed = 0
            num_points_unchanged = 0
            err_changed = 0
            err_unchanged = 0
            for states, actions, next_states in testloader:
            # for obs in testloader:
                # states, actions, next_states = obs['state'], obs['action'], obs['next_states']
                actions = actions.to(device, non_blocking=True)
                states = states.to(device, non_blocking=True)
                next_states = next_states.to(device, non_blocking=True)
                y_hat = cnn(states, actions)
                mask_changed = torch.ne(states, next_states)   # this will consider only points that change
                num_points_changed += mask_changed.sum().item()
                num_points_unchanged += (~mask_changed).sum().item()
            
                err_changed += (torch.ne(y_hat > 0, next_states) * mask_changed).sum().item()
                err_unchanged += (torch.ne(y_hat > 0, next_states) * (~mask_changed)).sum().item()

            print('\t unseen levels: err_changed: {}, err_unchanged: {}, num_changed: {}'.format(err_changed / num_points_changed,  err_unchanged / num_points_unchanged, num_points_changed))
            
            num_points_changed = 0
            num_points_unchanged = 0
            err_changed = 0
            err_unchanged = 0
            for states, actions, next_states in noveltyloader:
            # for obs in testloader:
                # states, actions, next_states = obs['state'], obs['action'], obs['next_states']
                actions = actions.to(device, non_blocking=True)
                states = states.to(device, non_blocking=True)
                next_states = next_states.to(device, non_blocking=True)
                y_hat = cnn(states, actions)
                mask_changed = torch.ne(states, next_states)   # this will consider only points that change
                num_points_changed += mask_changed.sum().item()
                num_points_unchanged += (~mask_changed).sum().item()
            
                err_changed += (torch.ne(y_hat > 0, next_states) * mask_changed).sum().item()
                err_unchanged += (torch.ne(y_hat > 0, next_states) * (~mask_changed)).sum().item()

            print('\t novelty levels: err_changed: {}, err_unchanged: {}, num_changed: {}'.format(err_changed / num_points_changed,  err_unchanged / num_points_unchanged, num_points_changed))
            





            cnn.train()

    torch.save(cnn.state_dict(), 'pretrained_model.pt')

