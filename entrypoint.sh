#!/bin/bash

# Start TensorBoard and Jupyter Notebook servers
LOGDIR=${TENSORBOARD_LOGDIR:-/app/ray_results}
mkdir -p "$LOGDIR"

sed -i "s/\"--bind_all\", default=True,/\"--bind_all\",/g" /usr/local/lib/python3.12/dist-packages/tensorboard/plugins/core/core_plugin.py
tensorboard --logdir "$LOGDIR" --host 0.0.0.0 --port 6006 &

jupyter lab \
    --ip=0.0.0.0 \
    --port=8888 \
    --no-browser \
    --allow-root \
    --NotebookApp.token='' \
    --NotebookApp.password=''