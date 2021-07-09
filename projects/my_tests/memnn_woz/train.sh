#!/bin/bash

parlai train_model  --candidates inline --eval-candidates inline  --eps 1000  -lr 0.01 --validation-every-n-epochs 1 --save-after-valid True --validation-patience 100 --ignore-bad-candidates True --batchsize 5 -m memnn -hops 3 --embedding-size 128 -t $1 -mf $2 

# --dynamic-batching full
# --candidates fixed --eval-candidates fixed  --fixed-candidates-path labels_full.dict