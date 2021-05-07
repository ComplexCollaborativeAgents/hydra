import torch
import torch.nn as nn


class MyCNN(nn.Module):
    def __init__(self, device='cpu'):
        super().__init__()

        self.conv0 = nn.Sequential(
            nn.Linear(84*48*13, 4096),
            nn.ReLU(),
        )
        self.conv1 = nn.Sequential(
            nn.Linear(4096, 2048),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(),
        )
        self.conv3 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU()
        )


        self.fc0 = nn.Sequential(
            nn.Linear(512, 2048),
            nn.ReLU()
        )

        self.dec = nn.Linear(2048, 2048, bias=True)
        self.x_enc = nn.Linear(2048, 2048, bias=False)
        self.u_enc = nn.Linear(5, 2048, bias=False)

        self.fc1 = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU()
        )

        self.deconv0 = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(),
        )

        self.deconv1 = nn.Sequential(
            nn.Linear(1024, 2048),
            nn.ReLU(),
        )

        self.deconv2 = nn.Sequential(
            nn.Linear(2048, 4096),
            nn.ReLU(),
        )

        self.deconv3 = nn.Sequential(
            nn.Linear(4096, 84*48*13),
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

        # x = self.fc0(x.squeeze())
        x = self.fc0(x)
        # x = (x.matmul(self.w_enc.t()) * u.matmul(self.w_u.t())).matmul(self.w_dec.t()) + self.b   # https://arxiv.org/pdf/1507.08750.pdf Sec 3.2
        x = self.dec(self.x_enc(x) * self.u_enc(u))   # https://arxiv.org/pdf/1507.08750.pdf Sec 3.2
        x = self.fc1(x)
        # x = x.view(*x.shape, 1, 1)
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
    trainval_set = ABDataset('../data_new_preproc', mode='train')
    test_set = ABDataset('../data_new_preproc', mode='test')
    novelty_set = ABDataset('../data_new_preproc', mode='Level_1_6')
    # torch_dataset = ABDataset('obs_data_preproc', transform)
    n_train = int(0.8 * len(trainval_set))
    n_val = len(trainval_set) - n_train
    train_set, val_set = random_split(trainval_set, [n_train, n_val])

    print(len(train_set), len(val_set), len(test_set), len(novelty_set))

    trainloader = DataLoader(
        train_set,
        batch_size= 32,
        shuffle=True,
        # num_workers=10,
        pin_memory=True,
        # collate_fn=collate_fn
    )

    valloader = DataLoader(
        val_set,
        batch_size=32,
        shuffle=True,
        # num_workers=10,
        pin_memory=True,
        # collate_fn=collate_fn
    )
    testloader = DataLoader(
        test_set,
        batch_size=32,
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
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(1000.),reduction='sum')
    optimizer = torch.optim.Adam(cnn.parameters(), 1e-4)


    for i in range(1000):
        print('Epoch: {}'.format(i))
        data_pnts = 0
        avg_loss = 0
        for states, actions, next_states in trainloader:
            actions = actions.to(device, non_blocking=True)
            states = states.to(device, non_blocking=True)
            next_states = next_states.to(device, non_blocking=True)
            states_resized = states.reshape(states.shape[0], -1)
            next_states_resized = next_states.reshape(next_states.shape[0], -1)

            y_hat = cnn(states_resized, actions)

            loss = loss_fn(y_hat, next_states_resized)
            loss = torch.mean(loss)
            data_pnts += states.shape[0]
            avg_loss+=loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print("AVG loss :", avg_loss/data_pnts)

        with torch.no_grad():
            cnn.eval()

            data_pnts = 0
            avg_loss = 0
            avg_acc = 0.
            avg_neg = 0.
            for states, actions, next_states in trainloader:
                actions = actions.to(device, non_blocking=True)
                states = states.to(device, non_blocking=True)
                next_states = next_states.to(device, non_blocking=True)
                states_resized = states.reshape(states.shape[0], -1)
                next_states_resized = next_states.reshape(next_states.shape[0], -1)
                y_hat = cnn(states_resized, actions)

                loss = loss_fn(y_hat, next_states_resized)
                data_pnts += states.shape[0]
                avg_loss+=loss
                avg_acc += ((y_hat > 0) == next_states_resized).sum()
                avg_neg += (1 - next_states_resized).sum()
            print("AVG TRAIN Loss :", avg_loss/data_pnts, "Acc: ", avg_acc / (data_pnts*48*84*13), "Neg: ", avg_neg / (data_pnts*48*84*13))

            data_pnts = 0
            avg_loss = 0
            avg_acc = 0.
            avg_neg = 0.
            for states, actions, next_states in valloader:
                actions = actions.to(device, non_blocking=True)
                states = states.to(device, non_blocking=True)
                next_states = next_states.to(device, non_blocking=True)
                states_resized = states.reshape(states.shape[0], -1)
                next_states_resized = next_states.reshape(next_states.shape[0], -1)
                y_hat = cnn(states_resized, actions)

                loss = loss_fn(y_hat, next_states_resized)
                data_pnts += states.shape[0]
                avg_loss+=loss
                avg_acc += ((y_hat > 0) == next_states_resized).sum()
                avg_neg += (1 - next_states_resized).sum()
            print("AVG VAL Loss :", avg_loss/data_pnts, "Acc: ", avg_acc / (data_pnts*48*84*13), "Neg: ", avg_neg / (data_pnts*48*84*13))

            data_pnts = 0
            avg_loss = 0
            avg_acc = 0
            avg_neg = 0.
            for states, actions, next_states in testloader:
                actions = actions.to(device, non_blocking=True)
                states = states.to(device, non_blocking=True)
                next_states = next_states.to(device, non_blocking=True)
                states_resized = states.reshape(states.shape[0], -1)
                next_states_resized = next_states.reshape(next_states.shape[0], -1)
                y_hat = cnn(states_resized, actions)
                loss = loss_fn(y_hat, next_states_resized)
                data_pnts += states.shape[0]
                avg_loss+=loss
                avg_acc += ((y_hat > 0) == next_states_resized).sum()
                avg_neg += (1 - next_states_resized).sum()
            print("AVG TEST LOSS :",avg_loss/data_pnts, "Acc: ", avg_acc / (data_pnts*48*84*13), "Neg: ", avg_neg / (data_pnts*48*84*13))



            cnn.train()

            torch.save(cnn.state_dict(), 'pretrained_model.pt')
