#!/bin/bash

session_name=parcel2d
# kill old session if exists
tmux kill-session -t "${session_name}"
sleep 1

# start new tmux session
tmux new-session -s "${session_name}" 'docker run -it --rm --name parcel2d \
--mount type=bind,source=${PWD}/data/backgrounds,target=/data/backgrounds \
--mount type=bind,source=${PWD}/data/objects,target=/data/objects \
--mount type=bind,source=${PWD}/data/distractors,target=/data/distractors \
--mount type=bind,source=${PWD},target=/app \
 generate_parcel2d:latest'