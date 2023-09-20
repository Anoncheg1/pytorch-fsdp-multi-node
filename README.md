# pytorch-fsdp-multi-node

Also, look at Parameter Server approach in Tensorflow [My repository with training in Parameter Server paradigm with Tensorflow](https://github.com/Anoncheg1/tensorflow-parameter-server)

Training ResNet50 in Fully Sharded Data Parallel (FSDP) approach at kubernetes cluster with help of Kubeflow.

Landmark 2020 is used, but you can switch to MNIST with commented lines.

ResNet is used, but you can switch to small "Net" by uncomment lines.

Validation and inference is not ready yet.

There is LandmarkDataset class that inherit torch.utils.data.Dataset.
This class have cache feature - it can save images after applying
"transform" to cache of pickle files and then read from this cache.

If you uncomment last line in main-dist-fsdp.py will be able to test code at one machine.


# files

- torch.yaml - Kubernetes yaml file.
- a.sh - used to gather stdout and stderr of main-dist-fsdp.py to grap logs and execute this file
- main-dist-fsdp.py - main file uses datasetmetareader.py and policies/
- main-local.py - simple PyTorch training with validation and inference, uses datasetmetareader.py
- datasetmetareader.py - function for prepareing Landmarks 2020 dataset
- policies/ - help classes for FSDP auto_wrap_policy parameter
