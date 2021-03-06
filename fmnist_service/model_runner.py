import asyncio
import torch
import functools

from fmnist_classifier.image_classifier import ImageClassifier

import logging 
logger = logging.getLogger(__name__)

class ModelRunner:
    """
    The Model Runner class is responsible for running a pretrained model on batched inputs coming in from a PubSub server
    To provide more throughput the process batching and inference is done asynchronously 
    The class should be run ModelRunner.__call__() in a non-blocking way so we can also listen on the PubSub server
    """
    def __init__(self, model_path, batch_size:int = 5, max_wait_time: int = 5,device: torch.device = torch.device('cpu')) -> None:
        self.loop = asyncio.get_event_loop()
        
        self.queue = []

        self.queue_lock = asyncio.Lock(loop=self.loop)
        self.needs_processing = asyncio.Event(loop=self.loop)

        self.device = device

        self.model = ImageClassifier.from_path(model_path)

        self.batch_size = batch_size
        self.max_wait_time = max_wait_time  

    def schedule_processing_if_needed(self):
        if len(self.queue) >= self.batch_size:
            self.needs_processing.set()

        elif len(self.queue) > 0:

            self.needs_processing_timer = self.loop.call_at(self.queue[0].time + self.max_wait_time, \
                                                            self.needs_processing.set)

    def run_model(self, data):
        return self.model.predict(data.to(self.device))

    async def __call__(self):
        logger.info('model runner loop started ...')

        while True:
            await self.needs_processing.wait()
            self.needs_processing.clear()

            if not self.queue: 
                 continue # fixes the timer issue 

            logger.info('started processing batch ...')

            if self.needs_processing_timer is not None:
                
                self.needs_processing_timer.cancel()
                self.needs_processing_timer = None


            async with self.queue_lock:

                batch = self.queue[:self.batch_size]
                del self.queue[:len(batch)]
                self.schedule_processing_if_needed()

            data_batch = torch.stack([req.data for req in batch], dim=0) # not good, timer freaks out
            data_batch.unsqueeze_(dim=1)
            

            result = await self.loop.run_in_executor(
                None, functools.partial(self.run_model, data_batch)
            )

            for item, r in zip(batch, result):
                item.result = r.item()
                item.done_event.set()

            logger.info('finished with processing batch ...')

            del batch

    async def process_request(self, data):
        req = Request(data, self.loop)
        logger.info(f'started on new request')

        async with self.queue_lock:
            self.queue.append(req)
            self.schedule_processing_if_needed()
        
        await req.done_event.wait()
        logger.info(f'finished request')
        return req.result




class Request:
    """
    Request is a helper class to use isntead of a dictionary 
    """
    def __init__(self, data, loop) -> None:
        
        self.time = loop.time()
        self.result = None
        self.done_event = asyncio.Event(loop=loop)
        self.data = data if isinstance(data, torch.Tensor) else torch.FloatTensor(data)
        
