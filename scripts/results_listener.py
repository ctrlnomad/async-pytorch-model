import logging

from pubsub import KafkaServer, GoogleServer
from scripts.train_fmnist import valid_ds

from argparse import ArgumentParser

logger = logging.getLogger(__name__)

import numpy as np


NUM_MSGS = len(valid_ds)
NUM_KEY_BYTES = np.log2(NUM_MSGS) // 8


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
    
    if args.gcloud:
        server = GoogleServer(project_id=args.project_id, 
                              request_topic_id=args.request_topic_id,
                              result_topic_id=args.result_topic_id,
                              subscription_id=args.subscription_id)

    else:
        server = KafkaServer(request_topic_id=args.request_topic_id, 
                             result_topic_id=args.result_topic_id, 
                             server_address=args.server_address)

    correct = 0
    try:
        for msg in server.sync_listen():
            result =  deserialise_key(msg.value) if isinstance(msg.value, bytes) else msg.value
            key = deserialise_key(msg.key) if isinstance(msg.key, bytes) else msg.key

            logger.info(f'msg with key={msg.key} came with result={result}')

            if valid_ds.targets[key].item() == result:
                correct += 1

    except KeyboardInterrupt:
        logger.info(f'VALID ACCURACY: [{correct/len(valid_ds):.3f}]')
    
    except Exception as e:
        logger.error(f'failed to consumer because: {e}')