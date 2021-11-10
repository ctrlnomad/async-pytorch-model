import asyncio, json
from typing import Generator

from numpy.core.arrayprint import array2string

from pubsub.base import PubSubServer
from pubsub.utils import array2bytes, bytes2array, async_proc
from kafka import KafkaProducer, KafkaConsumer
from aiokafka import AIOKafkaConsumer, AIOKafkaProducer

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class KafkaServer(PubSubServer):
    def __init__(self, consume_topic_id, server_address, loop, publish_topic_id=None) -> None:
        self.consume_topic_id = consume_topic_id
        self.publish_topic_id = publish_topic_id
        self.server_address = server_address

        self.producer = KafkaProducer(bootstrap_servers=[self.server_address])

        self.loop = loop


    async def _async_resolve(self, proc_fn):
        self.consumer = AIOKafkaConsumer(
                self.consume_topic_id, 
                bootstrap_servers=[self.server_address],
                loop=self.loop, value_deserializer=bytes2array)

        await self.consumer.start()
        tasks = []
        try:
            async for message in self.consumer:
                logger.info(f'received a new message')
                task = self.proc_and_send(proc_fn(message), key=message.key)
                self.loop.create_task(task)
                tasks.append(task)
        finally:
            await self.consumer.stop()

        await asyncio.gather(*tasks)

    async def proc_and_send(self, future, key=None):
        result = await future

        if not isinstance(result, bytes):
            result = result.to_bytes(2, byteorder='big')

        self.send(result, key=key)

    def start_transaction_service(self, proc_fn: callable):
        logging.info('started listening ... ')
        self.loop.run_until_complete(self._async_resolve(proc_fn))

    def send(self, msg, key=None):
        self.producer.send(self.publish_topic_id, msg, key=key)

    def sync_listen(self) -> Generator:
        consumer = KafkaConsumer(self.consume_topic_id)

        for msg in consumer:
            yield msg

        consumer.close()



if __name__ ==  '__main__':
    consume_topic_id = 'fmnist_request'
    publish_topic_id = 'fmnist_result'
    server_address = 'localhost:9092'

    loop = asyncio.get_event_loop()
    s = KafkaServer(consume_topic_id=consume_topic_id, 
                    publish_topic_id =publish_topic_id, 
                    server_address=server_address,
                    loop=loop)

    s.start_transaction_service(proc_fn=async_proc)
