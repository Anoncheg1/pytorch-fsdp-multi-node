# This is a sample Python script.
import pickle
import time
import logging
import os
from datetime import datetime
import torch
import torchvision.models as models
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.io import read_image
from torchvision import transforms
import numpy as np
from torch.utils.data import random_split
# - own
from datasetmetareader import get_dataset
# -- configure logger
logging.getLogger().setLevel(logging.INFO)

# -- config
IMG_WIDTH = 64
IMG_HEIGHT = 64
INPUT_SIZE = (IMG_HEIGHT, IMG_WIDTH, 3)
OUTPUT_SIZE = None  # let's get count of classes from Data
BATCH_SIZE = 25
EPOCHS = 1
DROUPOUT_RATE = 0.2
# OVERSAMPLING_ENABLED = False
# CLASS_WEIGHT_ENABLED = False
FLOAT16_FLAG = True
print("batch_size: " + str(BATCH_SIZE))
# - Set the device
device = "cuda" if torch.cuda.is_available() else "cpu"
# - Set the device globally
torch.set_default_device(device)

# If reserved memory is >> allocated memory try setting max_split_size_mb to avoid fragmentation.
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256"

if device == "cuda":
    GPU_SCORE = torch.cuda.get_device_capability()
    # GPU_SCORE = (8, 0)
    # optimization - perform faster matrix multiplications
    if GPU_SCORE >= (8, 0):
        print(f"[INFO] Using GPU with score: {GPU_SCORE}, enabling TensorFloat32 (TF32) computing (faster on new GPUs)")
        torch.backends.cuda.matmul.allow_tf32 = True
    else:
        print(f"[INFO] Using GPU with score: {GPU_SCORE}, TensorFloat32 (TF32) not available, to use it you need a GPU with score >= (8, 0)")
        torch.backends.cuda.matmul.allow_tf32 = False

default_float_dtype = torch.get_default_dtype()
if FLOAT16_FLAG:
    default_float_dtype = torch.float16


class LandmarkDataset(Dataset):
    def __init__(self, paths, labels, transform=None, target_transform=None, cache_path=None):
        """
        :cache_path - set if you want to create a cache or to load images from a cache
        """
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
        if not os.path.exists(save_path):
            raise Exception("no path")

        l = self.__len__()
        done_file_path = os.path.join(save_path, ".done-"+str(l))
        # - check if already saved
        # - TODO: chack that there is no other .done-???
        if os.path.exists(done_file_path):
            logging.info(f"Image's cache already exist, we will use {save_path}.")
            return
        logging.info(f"Saving image to cache folder {save_path}.")
        # - save files
        for i, img_path in enumerate(self.paths):
            if i % 10000 == 9999:
                logging.info(f"{i}, of, {l}")  # print steps
            image = read_image(img_path)
            if self.transform:
                image = self.transform(image)
            fp = os.path.join(save_path, str(i) + ".pkl")
            # torch.save(image, fp) # _use_new_zipfile_serialization = False
            with open(fp, 'wb') as f:
                pickle.dump(image, f)
        # - create file ".done-9999" to mark directory as full
        open(done_file_path, 'a').close()
            # torchvision.io.image.write_png(image,fp)
        self.cache_path = save_path

    def __getitem__(self, idx):
        if self.cache_path:
            # print("here", idx)
            # for i, img_path in enumerate(self.paths):
            fp = os.path.join(self.cache_path, str(idx) + ".pkl")
            # image = torch.load(fp) # , weights_only=True
            with open(fp, 'rb') as f:
                image = pickle.load(f)
                # image = torchvision.io.image.read_image(fp)
        else:
            image = read_image(self.paths[idx])
            if self.transform:
                image = self.transform(image)
        image = image.to(dtype=default_float_dtype).div(255)
        label = self.labels[idx]
        if self.target_transform:
            label = self.target_transform(label)
        # return image, label
        return image.to(device), torch.tensor(label, dtype=torch.long).to(device)


def train_one_epoch(epoch_index, training_loader, optimizer, model, loss_fn, tb_writer=None):
    """ training_loader is (inputs, labels) """
    model.train(True)

    running_loss = 0.
    # last_loss = 0.
    avg_loss = 0.
    correct = 0
    total = 0
    start_time = time.time()

    for i, data in enumerate(training_loader):

        inputs, labels = data

        optimizer.zero_grad()
        # -- forward, backward + optimize
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        # -- collect statistics
        # print("outputs", outputs.data)
        # print("outputs2", outputs.argmax(axis=1))
        # print("labels size", labels.size(0))
        total += labels.size(0)
        correct += (outputs.argmax(axis=1) == labels).sum().item()  # False, True -> count True, -> extract number
        # print("labels", labels)
        # print("total", total, "correct", correct)

        running_loss += loss.item()

        if i % 100 == 99:
            avg_loss = running_loss / i
            # - overwrite output:
            print(f'Batch {i + 1} loss: {round(avg_loss,2)}, accuracy raw: {correct / total}, time {time.time() - start_time} s')

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


def train_valid(model, training_loader, validation_loader, loss_fn, epochs):
    # timestamp = datetime.now().strftime('%Y%m%d_%H%M%S') # for saving checkpoints
    epoch_number = 0
    # optimizer = torch.optim.Adam(model.parameters())
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    for epoch in range(epochs):
        print('EPOCH {}:'.format(epoch_number + 1))
        # ---- train ----
        avg_loss = train_one_epoch(epoch_number,
                                   training_loader=training_loader,
                                   optimizer=optimizer,
                                   model=model,
                                   loss_fn=loss_fn,
                                   tb_writer=None)
        avg_vloss, acc = validate(model, validation_loader, loss_fn)
        print('Loss after epoch: train {} valid {} val_accuracy {}'.format(avg_loss, avg_vloss, acc))
        # if avg_vloss < best_vloss:
        #     best_vloss = avg_vloss
            # -- save checkpoint
            # model_path = 'model_{}_{}'.format(timestamp, epoch_number)
            # torch.save(model.state_dict(), model_path)  # save the model's state

        epoch_number += 1


def create_model(classes) -> torch.nn.Module:
    resnet = models.resnet50(weights=None)
    num_ftrs = resnet.fc.in_features
    resnet.fc = torch.nn.Linear(num_ftrs, out_features=classes)
    if FLOAT16_FLAG:
        resnet = resnet.half()
    return resnet


def main():
    logging.info("main")
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
    CACHE_DIR = "/tmp/c"
    # os.rmdir(CACHE_DIR) # if saving was not completed
    if not os.path.exists(CACHE_DIR):
        try:
            os.mkdir(CACHE_DIR)
        except:
            pass # noqa
    train_dataset = LandmarkDataset(x_train, y_train, transform=data_transform, cache_path=CACHE_DIR)
    valid_dataset: Dataset = LandmarkDataset(x_valid, y_valid, transform=data_transform)
    # train_val, _ = random_split(range(10), [0.1, 0.9], generator=torch.Generator(device=device).manual_seed(42))
    # -- DataLoader
    # from torch.utils.data.dataloader import default_collate
    generator = torch.Generator(device=device)
    train_loader: DataLoader = DataLoader(train_dataset,
                                          shuffle=True, batch_size=BATCH_SIZE,
                                          generator=generator)  # , pin_memory_device=device, pin_memory=True

    # collate_fn=lambda x: (default_collate(x[0]).to(device), default_collate(torch.from_numpy(x[1])).to(device))
    # train_val_loader: DataLoader = DataLoader(train_val, generator=generator)   # validate inside epoch
    valid_loader: DataLoader = DataLoader(valid_dataset, generator=generator)
    # -- train
    model: torch.nn.Module = create_model(OUTPUT_SIZE)  # load model definition
    print(model)
    # torch.cuda.empty_cache() # optimization - no change
    train_valid(model, training_loader=train_loader,
                validation_loader=valid_loader,
                loss_fn=torch.nn.CrossEntropyLoss(),
                epochs=EPOCHS)
    # -- save, load
    PATH = os.path.join(os.getcwd(), 'savedmodel')
    torch.save(model.state_dict(), PATH)
    model = create_model(OUTPUT_SIZE)
    model.load_state_dict(torch.load(PATH))
    # -- inference
    model.eval()
    img, lab = next(iter(DataLoader(valid_dataset, shuffle=True, batch_size=1
                                    ,generator=generator
    )))  # get random item
    print("lab", lab)
    result: torch.Tensor = model(img)

    print("result", np.argmax(result.cpu().detach().numpy()))


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
