#!/bin/bash

parlai train_model --candidates fixed --eval-candidates fixed  --fixed-candidates-path labels.dict --eps 1000 --dynamic-batching full  -lr 0.01 --validation-every-n-epochs 1 --save-after-valid True --validation-patience 100 --ignore-bad-candidates True -m memnn -hops 4 --embedding-size 128 -t woz:memnn_woz -mf $1 