#!/usr/bin/env bash
python \
/home/liuhy/Program/serving/tensorflow_serving/example/mnist_client.py \
--num_tests=1000 \
 --server=localhost:9000 \
 --concurrency=10 \
