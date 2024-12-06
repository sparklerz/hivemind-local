import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from tqdm.auto import tqdm
import time

import hivemind
import logging

# Set up logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger("Peer2")

# Create a parser to handle command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--initial_peers', type=str, default="/ip4/0.0.0.0/tcp/0/p2p/12D3KooWJFRJsgZG9TyCLXuLx9vH36bnpMxLaS9QpZ4exzsAPZBZ", help='Initial peers')
args = parser.parse_args()

# Create dataset and model, same as in the basic tutorial
# For this basic tutorial, we download only the training set
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

model = nn.Sequential(nn.Conv2d(3, 16, (5, 5)), nn.MaxPool2d(2, 2), nn.ReLU(),
                      nn.Conv2d(16, 32, (5, 5)), nn.MaxPool2d(2, 2), nn.ReLU(),
                      nn.Flatten(), nn.Linear(32 * 5 * 5, 10))
opt = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Create DHT: a decentralized key-value storage shared between peers
#dht = hivemind.DHT(initial_peers=['/ip4/127.0.0.1/tcp/COPY_FULL_ADDRESS_FROM_PEER1_OUTPUTS'], start=True)
dht = hivemind.DHT(
    host_maddrs=["/ip4/0.0.0.0/tcp/0"],
    initial_peers=[args.initial_peers],
    start=True)
print("To join the training, use initial_peers =", [str(addr) for addr in dht.get_visible_maddrs()])

# Track samples processed
samples_processed = 0
target_batch_size = 15

# Set up a decentralized optimizer that will average with peers in background
opt = hivemind.Optimizer(
    dht=dht,                  # use a DHT that is connected with other peers
    run_id='my_cifar_run',    # unique identifier of this collaborative run
    batch_size_per_step=3,   # each call to opt.step adds this many samples towards the next epoch
    target_batch_size=target_batch_size,  # after peers collectively process this many samples, average weights and begin the next epoch 
    optimizer=opt,            # wrap the SGD optimizer defined above
    use_local_updates=True,   # perform optimizer steps with local gradients, average parameters in background
    matchmaking_time=3.0,     # when averaging parameters, gather peers in background for up to this many seconds
    averaging_timeout=10.0,   # give up on averaging if not successful in this many seconds
    verbose=True              # print logs incessently
)

opt.load_state_from_peers()

# Note: if you intend to use GPU, switch to it only after the decentralized optimizer is created
with tqdm() as progressbar:
    while True:
        for x_batch, y_batch in torch.utils.data.DataLoader(trainset, shuffle=True, batch_size=32):
            opt.zero_grad()
            loss = F.cross_entropy(model(x_batch), y_batch)
            loss.backward()
            opt.step()

            samples_processed += len(x_batch)
            print(f"Peer2 processed {samples_processed} samples towards target {target_batch_size}")
            
            if samples_processed >= target_batch_size:
                print("Peer2 completed one averaging round")
                samples_processed = 0
            
            progressbar.desc = f"loss = {loss.item():.3f}"
            progressbar.update()
            print()
            time.sleep(5)