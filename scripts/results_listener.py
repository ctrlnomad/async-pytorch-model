import asyncio

import time, random, logging

from pubsub import KafkaServer, GoogleServer
from pubsub.utils import array2bytes
from classifier.train_fmnist import valid_ds


logger = logging.getLogger(__name__)

import numpy as np


NUM_MSGS = 10
NUM_KEY_BYTES = np.log2(NUM_MSGS) // 8
KAFKA = False

def seralise_key(key: int) -> bytes:
    return key.to_bytes(NUM_KEY_BYTES, byteorder='big')

def deserialise_key(key: bytes) -> int:
    return int.from_bytes(key, byteorder='big')

if __name__ ==  '__main__':
    loop = asyncio.get_event_loop()
    
    if KAFKA:
        publish_topic_id = 'fmnist_request'
        consume_topic_id = 'none'

        server_address = 'localhost:9092'

        server = KafkaServer(consume_topic_id=consume_topic_id, 
                        publish_topic_id =publish_topic_id, 
                        server_address=server_address,
                        loop=loop)
    else:
        project_id = 'vectorai-331519'
        topic_id = 'fmnist_results'
        subscription_id = 'fmnist_results_listener'

        server = GoogleServer(project_id=project_id, topic_id=topic_id, subscription_id=subscription_id)

    correct = 0
    
    for msg in server.sync_listen():
        msg.key