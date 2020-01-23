#!/usr/bin/env bash
tensorflow_model_server \
--enable_batching \
--port=9000 \
--model_name=mnist \
--model_base_path=/home/liuhy/tf_serving/mnist/monitored/ \
