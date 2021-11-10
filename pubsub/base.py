
from typing import Generator

from typing import Generator

class PubSubServer:

    def __init__(self, loop) -> None:
        pass

    def send(self, msg):
        raise NotImplementedError()

    def start_transaction_service(self, proc_fn: callable):
        raise NotImplementedError()

    def sync_listen(self) -> Generator:
        raise NotImplementedError()