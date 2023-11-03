#!/bin/bash
AMR_PATH=$1
SENT_PATH=$2
set -e -o pipefail

# postprocess var free amr so smatch can be calculated
postprocess_amr(){
	python3 /home/CE/mdare/code/AMR/postprocess_AMRs.py -f $AMR_PATH -s $SENT_PATH --force --threads 4
    rm ${AMR_PATH}.restore
    rm ${AMR_PATH}.restore.pruned
    rm ${AMR_PATH}.restore.coref
    rm ${AMR_PATH}.restore.wiki
    rm ${AMR_PATH}.restore.pruned.wiki
    rm ${AMR_PATH}.restore.pruned.wiki.coref.all
}

# Run tests here
postprocess_amr