import asyncio

import time, random, logging

from torch.types import Number

from pubsub import KafkaServer, GoogleServer
from pubsub.base import PubSubServer
from pubsub.utils import array2bytes
from scripts.train_classifier import parse_args, valid_ds

from argparse import ArgumentParser

logger = logging.getLogger(__name__)

import numpy as np


NUM_MSGS = len(valid_ds)
NUM_KEY_BYTES = int(np.ceil(np.log2(NUM_MSGS) / 8))

def seralise_key(key: int) -> bytes:
    return key.to_bytes(NUM_KEY_BYTES, byteorder='big')

def deserialise_key(key: bytes) -> int: 
    return int.from_bytes(key, byteorder='big')


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--gcloud', default=False)

    parser.add_argument('--project_id',  default='vectorai-331519')
    parser.add_argument('--subscription_id', default='fmnist_listener') # 'fmnist_results_listener'

    parser.add_argument('--save_path', default='./fmnist_network.pth')
    parser.add_argument('--request_topic_id', default='fmnist_request')
    parser.add_argument('--result_topic_id', default='fmnist_result')

    parser.add_argument('--server_address', default='localhost:9092')
    return parser.parse_args()

if __name__ ==  '__main__':
    args = parse_args()
    loop = asyncio.get_event_loop()

    server: PubSubServer
    
    if args.gcloud:
        server = GoogleServer(project_id=args.project_id, 
                              request_topic_id=args.request_topic_id,
                              result_topic_id=args.result_topic_id,
                              subscription_id=args.subscription_id, loop=loop)

    else:
        server = KafkaServer(request_topic_id=args.request_topic_id,
                            result_topic_id=args.result_topic_id, 
                            server_address=args.server_address, loop=loop)

    for i in range(NUM_MSGS):
        arr = valid_ds.data[i].cpu().numpy()
        img_bytes = array2bytes(arr)
        key_bytes = seralise_key(i)
        
        time.sleep(1e-2*random.random()) # simlate latency

        server.sync_send(img_bytes, key=key_bytes)
        logger.info(f'sent image #{i}')