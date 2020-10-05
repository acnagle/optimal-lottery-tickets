#!/bin/bash

python3 ../main.py --c 3 --epsilon 0.01 --hidden-size 500 --model "fc2" --target-net "../weights/fc2.pt" > "./fc2_approx.out"


# The following line will approximate a target network via subset-sum with the same settings defined above. The only difference is the
# --check_w_lt_eps argument, which thresholds all weight values below epsilon. In other words, any weight value in the target network 
# that is less than epsilon is approximated as 0. This can save a significant amount of time when approximating the entire network
# and it still performs well.
# python3 main.py --check_w_lt_eps --c 3 --epsilon 0.01 --hidden-size 500 --model "fc2" --target-net "./weights/fc2.pt" > "fc2_approx_check.out"

