# Hello! 👋

Each task has it's own folder:
 - Task One: `fmnist_classifier` directory
 - Task Two: `pubsub` directory
 - Task Three: `fmnist_service` directory

These folders contain the code for each task, however to run the service you will have to run one of the scripts in the `scripts` folder with a command like `python -m scripts.<script name here>`. Also please ensure you have all the packages from `requirements.txt` installed :)

Below I talk about each task in more detail.

## Task One

The `ImageClassifier` class has the code to train, save and load the model. It can also calculcate the validation accuracy.

To train the image classifier on a custom dataset please provide a new train and validation datasets as `torch.utils.data.Dataset` instances. When initialising `ImageClassifier` also please specify the new number of channels and output classes for your custom dataset. To make training on a new dataset easier I used `torch.LazyLinear` to automatically infer shape of input features. This feature is not production safe.

## Task Two
In the `pubsub` directory you will find the `base.py` file which outlines the functionality expected from a PubSub server. 

Then the `google_pubsub.py` and `kafka_pubsub.py` files use the appropriate client libraries to implement the functionality. The google production server works, however sometimes throws an exception because the asyncio loop gets nested, this can be patched by using the `nest_asyncio`. Using the kafka alternative is preferred.


## Task Three

Finally the `fmnsit_service` directory the `ModelRunner` class implements a batch-inference mechanism for asynchronously processing requests.

I used Chapter 15 from the `Deep Learning with PyTorch` book for reference here.

## Running The Whole Service

Provided we are using Kafka as our PubSub service and have `zookeeper` and `kafka-server` running we then:
 - run `python -m script.result_listener` (`Ctrl-C` will print the accuracy)
 - run `python -m script.run_classifier_service` in a separate terminal session
 - run `python -m script.send_images` in a separate terminal session


