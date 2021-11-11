# Hello! 👋

Each task has it's own folder:
 - Task One: `fmnist_classifier` directory
 - Task Two: `pubsub` directory
 - Task Three: `fmnist_service` directory

These folders contain the code for each task, however to run the service you will have to run one of the scripts in the `scripts` folder with a command like `python -m scripts.<script name here>`. Also please ensure you have all the packages from `requirements.txt` installed :)

Below I talk about each task in more detail.

## Task One

The `ImageClassifier` class has the code to train, save and load the model. It can also calculcate the validation accuracy.

To train the image classifier on a custom dataset please provide a new train and validation datasets as `torch.util.data.Dataset` instances. When initialising ImageClassifier also please specify the new number of channels and output classes for your custom dataset. To make training on a new dataset easier I used `torch.LazyLinear` to automatically infer shape of input features. This feature is not production safe.

## Task Two
In the `pubsub` directory includes the `base.py` file which outlines the functionality expected from a PubSub server. 

Then the `google_pubsub.py` and `kafka_pubsub.py` files use the appropriate client libraries to implement the functionality. The google production server works, however sometimes throws an exception because the asyncio loop gets nested sometimes, this can be patched by using the `nest_asyncio` module or simply use the kafka alternative. 


## Task Three

Finally the `fmnsit_service` directory the `Model Runner` class implements a batch-inference mechanism for asynchronously processing requests.


## Running The Whole Service

Provided we are using Kafka as our PubSub service and have `zookeeper` and `kafka-server` running we then:
 - run `python -m script.result_listener`
 - run `python -m script.run_classifier_service` in a separate terminal session
 - run `python -m script.send_images` in a separate terminal session


