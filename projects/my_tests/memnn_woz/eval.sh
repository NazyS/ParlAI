#!/bin/bash

# use multiprocessing_train for utilizing multiple gpus
parlai eval_model \
                --candidates fixed \
                --eval-candidates fixed \
                --interactive-candidates fixed \
                --fixed-candidates-path task5/candidates.txt \
                -lr 0.01 \
                --lr-scheduler fixed \
                --lr-scheduler-decay 1 \
                --optimizer sgd \
                --ignore-bad-candidates True \
                --batchsize 100 \
                -m memnn \
                -hops 3 \
                --embedding-size 32 \
                --task dialog_babi:task:5 \
                -mf task5-4/pos_enc/memnn_dialog_babi \
                --history-size 10 \
                --time-features False \
                --position-encoding True \
                --tensorboard-log True \
                # $@
                # --num-workers 4 \
                # --lr-scheduler cosine \
                # --max-train-steps 20000 \
                # --train-predict True \
                        # default is False
                # --dynamic-batching full \
                        # truncate is needed 
                # --fp16 True \
                # --fp16-impl mem_efficient \                        
                # --optimizer mem_eff_adam \                
                # --repeat-blocking-heuristic True \
                        # default is True
                # --memsize 32 \
                        # default is 32
                # --dict-file task5/dialog_babi_task5.dict \
                # -mf $1 \
                        # model file to save/init
                # --rank-top-k -1 \
                        # default is -1
                # --inference max \
                        # topk or max
                # --topk 5 
                # --return-cand-scores False \
                        # default is False                