import orjson
from collections import OrderedDict


class EasyDict(OrderedDict):
    """将字典形式改成"""
    __getattr__ = OrderedDict.get
    __setattr__ = OrderedDict.__setitem__
    __delattr__ = OrderedDict.__delitem__
    __eq__ = lambda self, other: self.id == other.id
    __hash__ = lambda self: self.id

    def __setitem__(self, key, value):
        value = self.convert_easy_dict(value)
        super().__setitem__(key, value)

    @staticmethod
    def convert_easy_dict(value):
        if (type(value) == dict) or (type(value) == OrderedDict):
            return EasyDict(value)
        elif isinstance(value, (list, tuple)):
            return [EasyDict.convert_easy_dict(v) for v in value]
        else:
            return value
    
    @staticmethod
    def convert_dict(info):
        data = orjson.dumps(info, option=orjson.OPT_APPEND_NEWLINE | orjson.OPT_SERIALIZE_NUMPY)
        return orjson.loads(data)


if __name__=="__main__":

    cfg = {"hello":111,
           "world":222,
           "test":{
               "ad":1,
               "type":[1,2,3, {"hee": 1}],
               "ccc":{"a":1}
           },
           "hooks": [{k:v for k, v in enumerate(range(5))}]
           }

    cfg = EasyDict(cfg)
    cfg.update({"heelo": 11, "he":{"ee":12}})

    for k, v in cfg.items():
        print(k, v)

    print(cfg.hello)
    print(cfg.world)
    print(cfg["hello"])
    cfg["world"] = 3234
    print(cfg.world)
    cfg["device"] = 0
    print(cfg.device)

    print(id(cfg.hello))
    print(id(cfg["hello"]))

    new_cfg = EasyDict.convert_dict(cfg)
    for k, v in cfg.items():
        print(k, v)
