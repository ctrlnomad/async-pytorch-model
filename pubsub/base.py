
from typing import Generator

class PubSubServer:
    """
    A base class for pubsub functionality, meant to be inherited. 
    Each PubSubServer should be able to act as a publisher or a subscriber through the sync_send() and sync_listen() methods.
    The PubSubServer should also be able to perform like a transactional service, like so:
        - listen on the request_topic_id
        - do some prcoessing asynchornously through the proc_fn passed in the start_transaction_service() method
        - publish the result to the result_topic_id
    """

    result_topic_id = None
    request_topic_id = None

    def __init__(self, *, loop=None) -> None:
        raise NotImplementedError()

    def sync_send(self, msg, key: bytes):
        raise NotImplementedError()
    
    def sync_listen(self) -> Generator:
        raise NotImplementedError()

    def start_transaction_service(self, proc_fn: callable):
        raise NotImplementedError()

