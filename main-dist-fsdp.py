import pickle
import time
import logging
import os
from datetime import datetime
from datetime import timedelta
import functools
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.io import read_image
from torchvision import transforms
import numpy as np
from torchvision import datasets

# - Set the device
device = "cuda" if torch.cuda.is_available() else "cpu"
logging.info("device = " + device)
# - Set the device globally
torch.set_default_device("cpu") # required for Shuffle
logging.info("default device = cpu")
# - distribute
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    CPUOffload,
    BackwardPrefetch,
)

import torch.cuda.nccl as nccl

# global flag that confirms ampere architecture, cuda version and
# nccl version to verify bfloat16 native support is ready

def bfloat_support():
    if device == "cpu":
        return False
    else:
        return (
        torch.version.cuda
        and torch.cuda.is_bf16_supported()
        # and packaging.version.parse(torch.version.cuda).release >= (11, 0)
        and dist.is_nccl_available()
        and nccl.version() >= (2, 10)
    )

from dataclasses import dataclass
from torch.distributed.fsdp import ShardingStrategy
from torch.distributed.fsdp.fully_sharded_data_parallel import StateDictType

import policies

# - own
from datasetmetareader import get_dataset

# -- FSDP
def setup(rank, world_size):
    # os.environ['MASTER_ADDR'] = 'localhost'
    # os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size,
                            timeout=timedelta(seconds=999999),
                            # init_method="file:///workspace/dist_test"
                            )
    # dist.init_process_group("gloo", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()

# -- configure logger
logging.getLogger().setLevel(logging.DEBUG)

# -- config
IMG_WIDTH = 64
IMG_HEIGHT = 64
INPUT_SIZE = (IMG_HEIGHT, IMG_WIDTH, 3)
OUTPUT_SIZE = None  # let's get count of classes from Data
BATCH_SIZE = 1
EPOCHS = 1
DROUPOUT_RATE = 0.2
# OVERSAMPLING_ENABLED = False
# CLASS_WEIGHT_ENABLED = False
FLOAT16_FLAG = False
CACHE_FLAG = False
logging.info("batch_size: " + str(BATCH_SIZE))

# If reserved memory is >> allocated memory try setting max_split_size_mb to avoid fragmentation.
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256"

# if device == "cuda":
#     GPU_SCORE = torch.cuda.get_device_capability()
#     # GPU_SCORE = (8, 0)
#     # optimization - perform faster matrix multiplications
#     if GPU_SCORE >= (8, 0):
#         print(f"[INFO] Using GPU with score: {GPU_SCORE}, enabling TensorFloat32 (TF32) computing (faster on new GPUs)")
#         torch.backends.cuda.matmul.allow_tf32 = True
#     else:
#         print(f"[INFO] Using GPU with score: {GPU_SCORE}, TensorFloat32 (TF32) not available, to use it you need a GPU with score >= (8, 0)")
#         torch.backends.cuda.matmul.allow_tf32 = False

default_float_dtype = torch.get_default_dtype()
if FLOAT16_FLAG:
    default_float_dtype = torch.float16


class LandmarkDataset(Dataset):
    def __init__(self, paths, labels, transform=None, target_transform=None, cache_path=None, get_first=None):
        """
        :cache_path - set if you want to create a cache or to load images from a cache
        """
        if get_first is not None:
            self.paths = paths[:get_first]
            self.labels = labels[:get_first]
        else:
            self.paths = paths
            self.labels = labels
        self.transform = transform
        self.target_transform = target_transform
        self.cache_path = cache_path
        if cache_path is not None:
            self.save_cached_images(cache_path)

    def __len__(self):
        return len(self.labels)

    def save_cached_images(self, save_path):
        sl = self.__len__()
        done_file_path = os.path.join(save_path, ".done-"+str(sl))
        # - check if already saved
        # - TODO: chack that there is no other .done-???
        if os.path.exists(done_file_path):
            logging.info(f"Image's cache already exist, we will use {save_path}.")
            return
        else:
            if not os.path.exists(save_path): # to be shure that folder exist
                os.mkdir(save_path)
                logging.info(f"Cache was not exist, we created new folder for it: {save_path}.")

        logging.info(f"Saving image to cache folder {save_path}.")
        # - save files
        for i, img_path in enumerate(sorted(self.paths)): # TODO: check that order is reproducible
            save_f_path = os.path.join(save_path, str(i) + ".pkl")
            if os.path.exists(save_f_path) and os.stat(save_f_path).st_size > 0:
                continue
            if i % 10000 == 9999:
                logging.info(f"{i}, of, {sl}")  # print steps
            image = read_image(img_path).to(device)
            if self.transform:
                image = self.transform(image)

            with open(save_f_path, 'wb') as f:
                pickle.dump(image, f)
        # - create file ".done-9999" to mark directory as full
        open(done_file_path, 'a').close()
        self.cache_path = save_path

    def __getitem__(self, idx):
        if self.cache_path:
            fp = os.path.join(self.cache_path, str(idx) + ".pkl")
            with open(fp, 'rb') as f:
                image = pickle.load(f)
        else:
            image = read_image(self.paths[idx])
            if self.transform:
                image = self.transform(image)
        image = image.to(dtype=default_float_dtype).div(255)
        label = self.labels[idx]
        if self.target_transform:
            label = self.target_transform(label)
        # return image.to(device), torch.tensor(label, dtype=torch.long).to(device)
        return image.to('cpu'), torch.tensor(label, dtype=torch.long).to('cpu')
        # return image, torch.tensor(label, dtype=torch.long)


def train_one_epoch(epoch_index, training_loader, optimizer, model, loss_fn, rank, tb_writer=None):
    """ training_loader is (inputs, labels) """
    model.train(True)
    ddp_loss = torch.zeros(2).to(device) # fsdp special

    if training_loader.sampler:
        logging.info(f"training_loader.sampler.set_epoch({epoch_index})")
        training_loader.sampler.set_epoch(epoch_index) # required for shuffle

    running_loss = 0.
    # last_loss = 0.
    avg_loss = 0.
    correct = 0
    total = 0
    start_time = time.time()

    for i, data in enumerate(training_loader):
        print("WTF", i, rank)
        logging.info(f"rank: {rank} , step: {i}, data[1].size: {data[1].size()}, {data[1]}")

        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        print("WTF1", i, rank)
        optimizer.zero_grad()
        print("WTF2", i, rank)
        # -- forward, backward + optimize
        outputs = model(inputs)
        print("WTF3", i, rank)
        loss = loss_fn(outputs, labels)
        print("WTF4", i, rank)
        loss.backward()
        print("WTF5", i, rank)
        optimizer.step()
        print("WTF6", i, rank)

        ddp_loss[0] += loss.item() # fsdp special
        ddp_loss[1] += len(data) # fsdp special
        # -- collect statistics
        total += labels.size(0)
        correct += (outputs.argmax(axis=1) == labels).sum().item()  # False, True -> count True, -> extract number

        running_loss += loss.item()

        if rank == 0 and i % 10 == 9:
            avg_loss = running_loss / i
            # - overwrite output:
            print(f'Batch {i + 1} loss: {round(avg_loss,2)}, accuracy raw: {correct / total}, time {time.time() - start_time} s')

    dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM) # fsdp special

    return avg_loss


def validate(model, validation_loader, loss_fn):
    model.eval()
    # ---- validate with validation_loader ----
    running_vloss = 0.0
    correct = 0
    total = 0

    # - Disable gradient computation and reduce memory consumption.
    with torch.no_grad():
        for i, vdata in enumerate(validation_loader):
            vinputs, vlabels = vdata
            voutputs = model(vinputs)
            vloss = loss_fn(voutputs, vlabels)
            running_vloss += vloss
            total += vlabels.size(0)
            correct += (voutputs.argmax(axis=1) == vlabels).sum().item()  # False, True -> count True, -> extract number

    return running_vloss / (i + 1), round(100 * correct / total, 2)


def train_valid(model, training_loader, validation_loader, loss_fn, epochs, rank=None):
    # timestamp = datetime.now().strftime('%Y%m%d_%H%M%S') # for saving checkpoints
    epoch_number = 0

    optimizer = optim.AdamW(model.parameters(), lr=0.002)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.85)

    for epoch in range(epochs):
        print('EPOCH {}:'.format(epoch_number + 1))
        # ---- train ----
        avg_loss = train_one_epoch(epoch_number,
                                   training_loader=training_loader,
                                   # optimizer=torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9),
                                   # optimizer=torch.optim.Adam(model.parameters()),
                                   optimizer=optimizer,
                                   model=model,
                                   loss_fn=loss_fn,
                                   rank=rank,
                                   tb_writer=None)
        avg_vloss, acc = validate(model, validation_loader, loss_fn)
        scheduler.step()
        print('Loss after epoch: train {} valid {} val_accuracy {}'.format(avg_loss, avg_vloss, acc))
        # if avg_vloss < best_vloss:
        #     best_vloss = avg_vloss
            # -- save checkpoint
            # model_path = 'model_{}_{}'.format(timestamp, epoch_number)
            # torch.save(model.state_dict(), model_path)  # save the model's state

        epoch_number += 1


class Net(nn.Module):
    """Small model for tests."""
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 1, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(900, 128)
        self.fc = nn.Linear(128, 10)

    def forward(self, x):

        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = nn.functional.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.dropout2(x)
        x = self.fc(x)
        output = nn.functional.log_softmax(x, dim=1)
        return output


def create_model(classes) -> torch.nn.Module:
    # model = Net()
    model = models.resnet50(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, out_features=classes)
    if FLOAT16_FLAG:
        model = model.half()
    logging.info(str(model))
    return model


def main(rank, world_size):
    logging.info("main")
    NUM_WORKERS = world_size - 1 # assume one master node
    setup(rank, world_size)  # FSDP - workers wait for master, master wait for worker
    logging.info("main after setup")
    x_train, x_valid, y_train, y_valid, OUTPUT_SIZE = get_dataset()
    print("classes: " + str(len(np.unique(y_train))))
    data_transform = transforms.Compose([
        transforms.RandomResizedCrop((IMG_HEIGHT, IMG_WIDTH), antialias=None),
        # transforms.ToTensor()  # to [0.0, 1.0]
        ])
    # -- save and load dataset --
    # - normal:
    # train_dataset = LandmarkDataset(x_train, y_train, transform=data_transform)
    # - with cache:
    if CACHE_FLAG:
        try:
            torch.multiprocessing.set_start_method('spawn',force=True)
            # torch.multiprocessing.set_start_method('fork',force=True)
        except RuntimeError:
            pass
    if CACHE_FLAG:
        train_dataset = LandmarkDataset(x_train, y_train, transform=data_transform, cache_path="/workspace/train_cache")
        valid_dataset = LandmarkDataset(x_valid, y_valid, transform=data_transform, cache_path="/workspace/test_cache")
    else:
        train_dataset = LandmarkDataset(x_train, y_train, transform=data_transform) # , get_first=1000
        valid_dataset = LandmarkDataset(x_valid, y_valid, transform=data_transform)
    # -- MNIST dataset (optional)
    # transform=transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.1307,), (0.3081,))
    # ])
    # train_dataset = datasets.MNIST('../data', train=True, download=True,
    #                     transform=transform)
    # valid_dataset = datasets.MNIST('../data', train=False,
    #                     transform=transform)
    train_sampler = DistributedSampler(train_dataset, rank=rank, num_replicas=world_size, shuffle=True)
    # train_sampler = torch.utils.data.SequentialSampler(train_dataset)
    valid_sampler = DistributedSampler(valid_dataset, rank=rank, num_replicas=world_size)

    # -- DataLoader
    print("device", device)
    train_kwargs = {'batch_size': BATCH_SIZE,
                    'sampler': train_sampler, # 'generator': torch.Generator(device='cpu') # - always, can not be altered
                    }
    test_kwargs = {'batch_size': 1,
                   'sampler': valid_sampler,
                   }
    cuda_kwargs = {'num_workers': NUM_WORKERS,
                   'pin_memory': True,
                   'pin_memory_device': device,
                   'shuffle': False}
    train_kwargs.update(cuda_kwargs)
    test_kwargs.update(cuda_kwargs)

    train_loader: DataLoader = DataLoader(train_dataset, **train_kwargs)
    valid_loader: DataLoader = DataLoader(valid_dataset, **test_kwargs)
    # -- FSDP requirements:
    # size_auto_wrap_policy = functools.partial(
    #     size_based_auto_wrap_policy, min_num_params=100
    # )
    # ???
    # init_start_event = torch.cuda.Event(enable_timing=True)
    # init_end_event = torch.cuda.Event(enable_timing=True)

    # -- test loader
    # print("train_loader", train_loader.generator.device)
    # train_sampler.set_epoch(0)
    # print(len(torch.randperm(len(train_dataset), generator=
    #                      torch.Generator()).tolist()))
    # img, lab = next(iter(train_loader))
    # print(img, lab)
    # print("dir(train_loader)", dir(train_loader))
    # print("cuda")
    # print(torch.cuda.is_available()) # True
    # print(torch.cuda.device_count()) # 1
    # print(torch.cuda.current_device()) # 0
    # print(torch.cuda.device(0)) # <torch.cuda.device at 0x7efce0b03be0>
    # print(torch.cuda.get_device_name(0)) # 'GeForce GTX 950M'
    # print()
    # -- train
    model: torch.nn.Module = create_model(OUTPUT_SIZE)  # load model definition
    model.to(device) # to GPU if exist
    # print(model)
    logging.info("Model is at divice:" + str( [ x.device for x in model.parameters()][0]))
    # - Apply FSDP wrapping to the model
    device_id = torch.cuda.current_device() if device == "cuda" else 0
    logging.info(f"device_id {device_id}")
    # model = FSDP(model, # T5 version
    #              my_auto_wrap_policy,
    #              auto_wrap_policy=t5_auto_wrap_policy,
    #              mixed_precision=mixed_precision_policy,
    #              sharding_strategy=fsdp_config.sharding_strategy,
    #              device_id=device_id,
    #              limit_all_gathers=fsdp_config.limit_all_gathers)
    model = FSDP(model,
                 sharding_strategy=ShardingStrategy.FULL_SHARD, # SHARD_GRAD_OP, # FULL_SHARD, # NO_SHARD,
                 # cpu_offload=CPUOffload(offload_params=True),
                 auto_wrap_policy=policies.wrapping.get_size_policy(300),
                 # backward_prefetch=torch.distributed.fsdp.BackwardPrefetch.BACKWARD_POST,
                 mixed_precision=False,
                 # ignored_modules=
                 device_id=device_id,
                 sync_module_states=False, # for  individually wrapped FSDP
                 forward_prefetch=True,
                 limit_all_gathers=False #True, #backward_prefetch

                 )

    # model = FSDP(model,
    # auto_wrap_policy=my_auto_wrap_policy,
    # cpu_offload=CPUOffload(offload_params=True))

    # print("model divice after FSDP", [ x.device for x in model.parameters()])

    # train_valid(model, training_loader=train_loader,
    #             validation_loader=valid_loader,
    #             loss_fn=torch.nn.CrossEntropyLoss(),
    #             epochs=EPOCHS, rank=rank)


    #
    # # -- save, load
    # PATH = os.path.join(os.getcwd(), 'savedmodel')
    # torch.save(model.state_dict(), PATH)
    # model = create_model(OUTPUT_SIZE)
    # model.load_state_dict(torch.load(PATH))
    # # -- inference
    # model.eval()
    # img, lab = next(iter(DataLoader(valid_dataset, shuffle=True, batch_size=1
    #                                 ,generator=generator
    # )))  # get random item
    # print("lab", lab)
    # result: torch.Tensor = model(img)
    #
    # print("result", np.argmax(result.cpu().detach().numpy()))

    dist.barrier()
    cleanup()


if __name__ == '__main__':
    # -- rang = rank of the worker within a worker group
    # -- world_size = The total number of workers in a worker group.
    WORLD_SIZE = int(os.environ.get('WORLD_SIZE', 1))
    RANK = int(os.environ.get('RANK', 0))
    logging.info(f"WORLD_SIZE={WORLD_SIZE}, RANK = {RANK}")
    if RANK == 0:
        main(rank=RANK, world_size=WORLD_SIZE)
    else: # workers is looped (not working properly)
        while True:
            try:
                main(rank=RANK, world_size=WORLD_SIZE)
            finally:
                time.sleep(3)
                dist.barrier()
                cleanup()
    # main(rank=0, world_size=1) # test with run at master only
