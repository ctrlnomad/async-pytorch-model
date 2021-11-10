from concurrent.futures import TimeoutError
from functools import partial
from typing import Generator
from google.cloud import pubsub_v1
from google.api_core import retry
import google.cloud.pubsub_v1.subscriber.message as ≈

from pubsub.base import PubSubServer
import time
import asyncio

class GoogleServer(PubSubServer):

    def __init__(self, project_id, topic_id, subscription_id) -> None:

        self.publisher = pubsub_v1.PublisherClient()
        self.subscriber = pubsub_v1.SubscriberClient()

        self.topic_path = self.publisher.topic_path(project_id, topic_id)
        self.subscription_path = self.subscriber.subscription_path(project_id, subscription_id)


        self.timeout = None

    def send(self, msg, key=None):
        return self.publisher.publish(self.topic_path, msg, attrs={'key': key}).result()        


    def start_transaction_service(self, fn):

        future = self.subscriber.subscribe(self.subscription_path, callback=partial(self.callback_and_send, fn=fn))

        with self.subscriber:
            try:
                future.result(timeout=self.timeout)
            except TimeoutError:
                future.cancel()  # Trigger the shutdown.
                future.result()  # Block until the shutdown is complete.

            

    def callback_and_send(self, msg,  fn: callable) -> None:
        print(f"Received {msg.data}.")
        try:
            result = asyncio.wait(fn(msg.data))
            msg.ack()
            self.send(result, key=msg.attrs['key'])

        except Exception as e:
            print(f'failed: {e}')


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