#!/usr/bin/env sh
# Compute the mean image from the imagenet training lmdb
# N.B. this is available in data/ilsvrc12

EXAMPLE=/gpu1
DATA=/gpu1
TOOLS=../../../build/tools

$TOOLS/compute_image_mean $EXAMPLE/train_lmdb \
  $DATA/image_mean.binaryproto

echo "Done."
