from braindecode.datasets import MOABBDataset
from TFSICNet import MergedModel

subject_id = 3
dataset = MOABBDataset(dataset_name="BNCI2015_001", subject_ids=[3])

import numpy as np

from braindecode.preprocessing import (
    exponential_moving_standardize,
    preprocess,
    Preprocessor,
)

low_cut_hz = 4.0  # low cut frequency for filtering
high_cut_hz = 38.0  # high cut frequency for filtering
# Parameters for exponential moving standardization
factor_new = 1e-3
init_block_size = 1000

transforms = [
    Preprocessor("pick_types", eeg=True, meg=False, stim=False),  # Keep EEG sensors
    # Preprocessor(
    #     lambda data, factor: np.multiply(data, factor),  # Convert from V to uV
    #     factor=1e6,
    # ),
    # Preprocessor("filter", l_freq=low_cut_hz, h_freq=high_cut_hz),  # Bandpass filter
    Preprocessor(
        exponential_moving_standardize,  # Exponential moving standardization
        factor_new=factor_new,
        init_block_size=init_block_size,
    ),
]

# Transform the data
preprocess(dataset, transforms, n_jobs=-1)

from braindecode.preprocessing import create_windows_from_events

trial_start_offset_seconds = -0.5
# Extract sampling frequency, check that they are same in all datasets
sfreq = dataset.datasets[0].raw.info["sfreq"]
assert all([ds.raw.info["sfreq"] == sfreq for ds in dataset.datasets])
# Calculate the trial start offset in samples.
trial_start_offset_samples = int(trial_start_offset_seconds * sfreq)

# Create windows using braindecode function for this. It needs parameters to define how
# trials should be used.
windows_dataset = create_windows_from_events(
    dataset,
    trial_start_offset_samples=trial_start_offset_samples,
    trial_stop_offset_samples=0,
    preload=True,
)
print(windows_dataset.description)
import torch
from braindecode.models import ShallowFBCSPNet, EEGNetv4, ATCNet
from braindecode.util import set_random_seeds

cuda = torch.cuda.is_available()  # check if GPU is available, if True chooses to use it
device = "cuda" if cuda else "cpu"
if cuda:
    torch.backends.cudnn.benchmark = True
seed = 20200220
set_random_seeds(seed=seed, cuda=cuda)

n_classes = 2
classes = list(range(n_classes))
# Extract number of chans and time steps from dataset
n_channels = windows_dataset[0][0].shape[0]
input_window_samples = windows_dataset[0][0].shape[1]

# The ShallowFBCSPNet is a `nn.Sequential` model

# model = EEGNetv4(
#     n_channels,
#     n_classes,
#     input_window_samples,
#     #final_conv_length="auto",
# )

model = MergedModel(
    n_channels,
    n_classes,
    input_window_size=input_window_samples,
    # final_conv_length="auto",
)

# model = ATCNet(n_channels, n_classes,
#                                          add_log_softmax=False,input_window_seconds=4.5,sfreq=250,conv_block_pool_size_1=8,conv_block_pool_size_2=7)

# Display torchinfo table describing the model
print(model)

# Send model to GPU
if cuda:
    model.cuda()

splitted = windows_dataset.split("session")
train_set = splitted['0A']  # Session train
test_set = splitted['1B']  # Session evaluation

from torch.nn import Module
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

lr = 0.001
weight_decay = 0
batch_size = 40
n_epochs = 200

from tqdm import tqdm


# Define a method for training one epoch


def train_one_epoch(
        dataloader: DataLoader, model: Module, loss_fn, optimizer,
        scheduler: CosineAnnealingLR, epoch: int, device, print_batch_stats=True
):
    model.train()  # Set the model to training mode
    train_loss, correct = 0, 0

    progress_bar = tqdm(enumerate(dataloader), total=len(dataloader),
                        disable=not print_batch_stats)
    for batch_idx, (X, y, _) in progress_bar:
        X, y = X.to(device), y.to(device)
        # if  y==0 or y==1 or y==2 or y==3:
        #     # count+=1
        #     # print(count)
        #     continue
        optimizer.zero_grad()
        pred, weights, weight_s = model(X)
        # print(weight_s.shape)
        # pred,weights,weight_s = model(X)
        loss = loss_fn(pred, y)
        loss.requires_grad_(True)
        loss.backward()
        optimizer.step()  # update the model weights
        optimizer.zero_grad()

        train_loss += loss.item()
        correct += (pred.argmax(1) == y).sum().item()

        if print_batch_stats:
            progress_bar.set_description(
                f"Epoch {epoch}/{n_epochs}, "
                f"Batch {batch_idx + 1}/{len(dataloader)}, "
                f"Loss: {loss.item():.6f}"
                # f"LR: {scheduler.get_lr():.6f}"
            )
    # Update the learning rate
    scheduler.step()
    correct /= len(dataloader.dataset)
    return train_loss / len(dataloader), correct


@torch.no_grad()
def test_model(
        dataloader: DataLoader, model: Module, loss_fn, print_batch_stats=True, count=1
):
    size = len(dataloader.dataset)
    n_batches = len(dataloader)
    model.eval()  # Switch to evaluation mode
    test_loss, correct = 0, 0

    if print_batch_stats:
        progress_bar = tqdm(enumerate(dataloader), total=len(dataloader))
    else:
        progress_bar = enumerate(dataloader)
    IF = 0
    for batch_idx, (X, y, _) in progress_bar:
        X, y = X.to(device), y.to(device)

        # pred,weights,weight_s = model(X)
        pred, weights, weight_s = model(X)

        sum = 0

        if count == 1:  # and IF==0:
            # IF=1
            # print('jjj')
            if IF == 0:
                IF = 1
                with open('weights_g.txt', 'a') as f:

                    tensor = weight_s.to('cpu')


                    numpy_array = tensor.numpy()
                    # f.write(str(abs(numpy_array)))
                    # for xx in range(72):
                    for k in range(numpy_array.shape[0]):
                        # print(y[k])
                        # if y[k]==3:
                        for i in range(numpy_array.shape[1]):
                            # sum+=numpy_array[i][k]
                            # if i==numpy_array.shape[1]-1:
                            #     f.write(str(sum)+' ')
                            #     sum=0
                            for j in range(numpy_array.shape[2]):
                                sum += numpy_array[k][j][i]
                                if j == numpy_array.shape[2] - 1:
                                    f.write(str(sum) + ' ')
                                    sum = 0

                        f.write('\n')
                    f.close()
            with open('weights_f.txt', 'a') as f:

                tensor = weights.to('cpu')


                numpy_array = tensor.numpy()
                # f.write(str(abs(numpy_array)))
                # for xx in range(72):
                for k in range(numpy_array.shape[0]):
                    # print(y[k])
                    if y[k] == 1:
                        print(y[k], k)
                        for i in range(numpy_array.shape[1]):
                            # sum+=numpy_array[i][k]
                            # if i==numpy_array.shape[1]-1:
                            #     f.write(str(sum)+' ')
                            #     sum=0
                            for j in range(numpy_array.shape[2]):
                                sum += numpy_array[k][j][i]
                                if j == numpy_array.shape[2] - 1:
                                    f.write(str(sum) + ' ')
                                    sum = 0

                        f.write('\n')
                f.close()

        batch_loss = loss_fn(pred, y).item()

        test_loss += batch_loss
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        if print_batch_stats:
            progress_bar.set_description(
                f"Batch {batch_idx + 1}/{len(dataloader)}, "
                f"Loss: {batch_loss:.6f}"
            )

    test_loss /= n_batches
    correct /= size

    print(
        f"Test Accuracy: {100 * correct:.1f}%, Test Loss: {test_loss:.6f}\n"
    )
    return test_loss, correct


# Define the optimization
optimizer = torch.optim.Adam(model.parameters(),
                             lr=lr, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=199, eta_min=0.000001)
# Define the loss function
# We used the NNLoss function, which expects log probabilities as input
# (which is the case for our model output)
loss_fn = torch.nn.CrossEntropyLoss()

# train_set and test_set are instances of torch Datasets, and can seamlessly be
# wrapped in data loaders.
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(train_set, batch_size=40)
count = 0
for epoch in range(1, n_epochs + 1):
    print(f"Epoch {epoch}/{n_epochs}: ", end="")
    model.train()
    train_loss, train_accuracy = train_one_epoch(
        train_loader, model, loss_fn, optimizer, scheduler, epoch, device,
    )
    if epoch == n_epochs:
        count = 1
    model.eval()
    test_loss, test_accuracy = test_model(test_loader, model, loss_fn, count=count)

    print(
        f"Train Accuracy: {100 * train_accuracy:.2f}%, "
        f"Average Train Loss: {train_loss:.6f}, "
        f"Test Accuracy: {100 * test_accuracy:.1f}%, "
        f"Average Test Loss: {test_loss:.6f}\n"
    )
