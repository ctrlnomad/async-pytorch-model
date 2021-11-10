
import asyncio
import nest_asyncio # i don't like this
nest_asyncio.apply()

from pubsub import KafkaServer, GoogleServer

from classifier.image_classifier import ImageClassifier
from fmnist_service.model_runner import ModelRunner

from argparse import ArgumentParser

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--save_path', required=True)
    parser.add_argument('--topic_id', required=True)

    parser.add_argument('--server_address', default='localhost:9092')
    parser.add_argument('--project_id', default=None)
    
    return parser.parse_args()

if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    save_path = './fmnist_network.pth'
    # KAFKA CONFIG
    # consume_topic_id = 'fmnist_request'
    # publish_topic_id = 'fmnist_result'
    # server_address = 'localhost:9092'
    
    # server = KafkaServer(consume_topic_id=consume_topic_id, 
    #                      publish_topic_id=publish_topic_id, 
    #                      server_address=server_address,
    #                      loop=loop)

    project_id = 'vectorai-331519'

    request_topic_id = 'fmnist_requests'
    result_topic_id = 'fmnist_results'

    subscription_id = 'fmnist_listener'


    server = GoogleServer(project_id=project_id, request_topic_id=request_topic_id,
                        result_topic_id=result_topic_id, subscription_id=subscription_id, loop=loop)

    classifier = ImageClassifier.from_path(save_path)

    runner = ModelRunner(save_path)
    task = loop.create_task(runner())

    server.start_transaction_service(runner.process_request)
    asyncio.wait_for(task)




