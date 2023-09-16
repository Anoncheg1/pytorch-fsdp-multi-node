# pytorch-fsdp-multi-node

Training ResNet50 in Fully Sharded Data Parallel (FSDP) approach at kubernetes cluster with help of Kubeflow.

Landmark 2020 is used, but you can switch to MNIST with commented lines.

ResNet is used, but you can switch to small "Net" by uncomment lines.

Validation and inference is not ready yet.

Also look at [Link to my repository with training in Parameter Server paradigm with Tensorflow](https://github.com/Anoncheg1/tensorflow-parameter-server)

# files

- torch.yaml - Kubernetes yaml file.
- a.sh - used to gather stdout and stderr of main-dist-fsdp.py to logs and execute this file
- main-dist-fsdp.py - main file uses datasetmetareader.py and policies/
- main-local.py - simple PyTorch training with validation and inference, uses datasetmetareader.py
- datasetmetareader.py - function for prepareing Landmarks 2020 dataset
- policies/ - help classes for FSDP auto_wrap_policy parameter
