from models.updown import UpDown
from models.xlan import XLAN
from models.xtransformer import XTransformer

__factory = {
    'XLAN': XLAN,
    'XTransformer': XTransformer
}

def names():
    return sorted(__factory.keys())

def create(name, *args, **kwargs):
    if name not in __factory:
        raise KeyError("Unknown caption model:", name)
    return __factory[name](*args, **kwargs)