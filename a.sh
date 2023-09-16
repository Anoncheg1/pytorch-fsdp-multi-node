#!/bin/sh
# apt install -y moreutils
#set +m # disable job control in order to allow lastpipe
#shopt -s lastpipe
# set -u # Report Non-Existent Variables
# set -e # It terminates the execution when the error occurs. (does not work with piped commands. use Set -eo pipefail)

# pip3 install pandas scikit-learn

# - TFJob:
# who=$(python -c "import tensorflow as tf ; v=tf.distribute.cluster_resolver.TFConfigClusterResolver() ; print(v.task_type, v.task_id)" | tr " " "_")
# - PyTorchJob:
who=$(echo $HOSTNAME | cut -d '-' -f 3,4)
# if [[ "$who" == "master-0" ]];then
#     rm dist_test
# fi
rm -f logs/${who}.log &>/dev/null
mkdir logs &>/dev/null
echo WHO=${who}


pgrep python | xargs kill -s 9 &>/dev/null
if [ "$1" = "kill" ]; then return ; fi

set -o pipefail # exit execution if one of the commands in the pipe fails.

script -c 'python main-dist-fsdp.py' |& sed -u "s/^/${who}: /" |& perl -ne 'use IO::Handle ; printf "%s %s",  scalar time(), $_ ; STDOUT->autoflush(1) ;' |& tee -a logs/${who}.log
