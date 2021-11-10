import asyncio, io, time
from PIL import Image
import numpy as np
import base64
import json


import logging
logger = logging.getLogger(__name__)

async def async_proc(msg):
    logger.info(f'got array shape={msg.value.shape}')
    logger.info(f'started processing msg with key={msg.key}')
    await asyncio.sleep(5)
    logger.info(f'finished processing msg with key={msg.key}')
    return msg.key


def proc(msg):
    logger.info(f'started processing msg with key={msg.key}')
    time.sleep(5)
    logger.info(f'finished processing msg with key={msg.key}')
    return msg.key



def array2bytes(numpy_array:np.array) -> str:
    # Create a Byte Stream Pointer
    bytesio = io.BytesIO()
    
    # Use PIL JPEG reduction to save the image to bytes
    Image.fromarray(numpy_array).save(bytesio, format="JPEG")
    
    # Set index to start position
    bytesio.seek(0)
    return bytesio.getvalue()



def bytes2array(bs) -> np.array:

    # Read byte array to an Image
    bs = io.BytesIO(bs)
    bs.seek(0)
    im = Image.open(bs)
    
    # Return Image to numpy array format
    return np.array(im)