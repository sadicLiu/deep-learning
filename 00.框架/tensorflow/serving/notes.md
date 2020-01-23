# TF Serving

## Serving a TensorFlow Model(Basic)

1. Each version sub-directory contains the following files:
   - saved_model.pb is the serialized tensorflow::SavedModel. It includes one or more graph definitions of the model, as well as metadata of the model such as signatures.
   - variables are files that hold the serialized variables of the graphs.
  
    ```txt
      1
      ├── saved_model.pb
      └── variables
          ├── variables.data-00000-of-00001
          └── variables.index

    ```
2. Load Exported Model With Standard TensorFlow ModelServer  
  `tensorflow_model_server --port=9000 --model_name=mnist --model_base_path=/tmp/mnist_model/`
3. Test The Server  
  `python tensorflow_serving/example/mnist_client.py --num_tests=1000 --server=localhost:9000`

## Building Standard TensorFlow ModelServer

> Build the standard TensorFlow ModelServer that dynamically discovers and serves new versions of a trained TensorFlow model

1. Two parts:  
    - mnist_saved_model.py
    - main.cc
2. Steps:  
    1. Train (with 100 iterations) and export the first version of model  
      `/home/liuhy/Program/serving/tensorflow_serving/example/scripts/mnist_train_export_100.sh`
    2. Train (with 2000 iterations) and export the second version of model  
      `/home/liuhy/Program/serving/tensorflow_serving/example/scripts/mnist_train_export_2000.sh`
    3. Copy the first version of the export to the monitored folder and start the server to test
    4. Copy the second version of the export to the monitored folder and start the server to test
    5. You'll find that the results are different, indicates that you have used two versions of models

## SignatureDefs in SavedModel for TensorFlow Serving

1. A SignatureDef requires specification of:  
    - inputs as a map of string to TensorInfo.
    - outputs as a map of string to TensorInfo.
    - method_name (which corresponds to a supported method name in the loading tool/system).
    > Note that TensorInfo itself requires specification of name, dtype and tensor shape. While tensor information is already present in the graph, it is useful to explicitly have the TensorInfo defined as part of the SignatureDef since tools can then perform signature validation, etc. without having to read the graph definition.
2. TensorFlow Serving provides high level APIs for performing inference. To enable these APIs, models must include one or more SignatureDefs that define the exact TensorFlow nodes to use for input and output.