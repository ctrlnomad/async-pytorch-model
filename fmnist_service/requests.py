
import json

class Request:
    def __init__(self, key: int) -> None:
        self.key = key
        pass
    
    def to_bytes(self, img_bytes):
        d = {
            'key' : self.key,
            'img_bytes': img_bytes,
        }
        return json.dumps(d).encode('utf-8')