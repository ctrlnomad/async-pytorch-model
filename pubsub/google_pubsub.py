from concurrent.futures import TimeoutError
from functools import partial

from typing import Generator
from google.cloud import pubsub_v1
from google.api_core import retry


from pubsub.base import PubSubServer
from pubsub.utils import bytes2array
import asyncio

import logging
logger = logging.getLogger(__name__)

class GoogleServer(PubSubServer):

    def __init__(self, project_id, request_topic_id, result_topic_id, subscription_id , loop=None) -> None:

        self.publisher = pubsub_v1.PublisherClient()
        self.subscriber = pubsub_v1.SubscriberClient()

        self.request_topic_path = self.publisher.topic_path(project_id, request_topic_id)
        self.result_topic_path = self.publisher.topic_path(project_id, result_topic_id)

        self.subscription_path = self.subscriber.subscription_path(project_id, subscription_id)

        self.loop = loop
        self.timeout = None

    def send(self, msg, key: int=None, topic=None ):
        if topic is None: 
            topic = self.request_topic_path

        return self.publisher.publish(topic, msg, key=str(key), timeout=self.timeout).result()        


    def start_transaction_service(self, fn):
        logger.info(f"transaction service started listening ...")

        future = self.subscriber.subscribe(self.subscription_path, callback=partial(self.callback_and_send, fn=fn))

        with self.subscriber:
            try:
                future.result(timeout=self.timeout)
            except TimeoutError:
                future.cancel()  # Trigger the shutdown.
                future.result()  # Block until the shutdown is complete.

            

    def callback_and_send(self, msg,  fn: callable) -> None:

        try:
            key = msg.attributes['key']
            logger.info(f"received msg with key {key} ...")
            msg.ack()
            logger.info('msg acknowledged')

            logger.info(f'before msg data: {type(msg.data)} {len(msg.data)}')
            msg_data = bytes2array(msg.data) 
            logger.info(f'after msg data: {type(msg_data)} {len(msg_data)}')

            result = self.loop.run_until_complete(fn(msg_data, key))
            logger.info(f'result for msg with key [{key}] is [{result}]')

            if not isinstance(result, bytes):
                result = result.to_bytes(2, byteorder='big')

            self.send(result, topic=self.result_topic_path, key=key) # different 

        except Exception as e:
            logger.info(f"failed to process {msg} becasue {e}")


    def sync_listen(self, num_messages=1e3) -> Generator:

        with self.subscriber:

            response = self.subscriber.pull(
                request={"subscription": self.subscription_path, 
                "max_messages": num_messages},
                retry=retry.Retry(deadline=300),
            )

            ack_ids = []
            for received_message in response.received_messages:
                yield received_message
                ack_ids.append(received_message.ack_id)


            self.subscriber.acknowledge(
                request={"subscription": self.subscription_path, "ack_ids": ack_ids}
            )