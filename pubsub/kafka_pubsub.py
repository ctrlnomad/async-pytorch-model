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
    def __init__(self, request_topic_id, result_topic_id, server_address, loop=None) -> None:
        self.request_topic_id = request_topic_id
        self.result_topic_id = result_topic_id

        self.server_address = server_address

        self.producer = KafkaProducer(bootstrap_servers=[self.server_address])

        self.loop = loop


    async def _async_resolve(self, proc_fn):
        self.consumer = AIOKafkaConsumer(
                self.request_topic_id, 
                bootstrap_servers=[self.server_address],
                loop=self.loop, value_deserializer=bytes2array)

        await self.consumer.start()
        tasks = []
        try:
            async for message in self.consumer:
                logger.info(f'received a new message')
                task = self.proc_and_send(proc_fn(message.value), key=message.key)
                self.loop.create_task(task)
                tasks.append(task)
        finally:
            await self.consumer.stop()

        await asyncio.gather(*tasks)

    async def proc_and_send(self, future, key=None):
        result = await future

        if not isinstance(result, bytes):
            result = result.to_bytes(2, byteorder='big')

        self.sync_send(result, key, topic=self.result_topic_id)

    def start_transaction_service(self, proc_fn: callable):
        logging.info('started listening ... ')
        self.loop.run_until_complete(self._async_resolve(proc_fn))

    def sync_send(self, msg, key: bytes, topic = None):
        if topic is None:
            topic = self.request_topic_id

        self.producer.send(topic, msg, key=key)
        logger.info(f'sent a msg with key [{key}] to [{topic}]')

    def sync_listen(self, topic=None) -> Generator:
        if topic is None:
            topic = self.result_topic_id

        logger.info(f'started sync listening on [{topic}]')
        consumer = KafkaConsumer(topic)

        for msg in consumer:
            yield msg

        consumer.close()


