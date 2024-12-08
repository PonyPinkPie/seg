# adapted from https://github.com/open-mmlab/mmcv
import collections
import os.path as osp
import sys
# from addict import Dict
from argparse import ArgumentParser
from importlib import import_module
from .io import load_json


# try:
#     if sys.version_info < (3, 3):
#         Sequence = collections.Sequence
#         Iterable = collections.Iterable
#     else:
#         Sequence = collections.abc.Sequence
#         Iterable = collections.abc.Iterable
# except:
#     Sequence = collections._collections_abc.Sequence
#     Iterable = collections._collections_abc.Iterable
#
#
# class ConfigDict(Dict):
#     def __missing__(self, name):
#         raise KeyError(name)
#
#     def __getattr__(self, name):
#         try:
#             value = super(ConfigDict, self).__getattr__(name)
#         except KeyError:
#             ex = AttributeError("'{}' object has no attribute '{}'".format(
#                 self.__class__.__name__, name))
#         except Exception as e:
#             ex = e
#         else:
#             return value
#         raise ex
#
#
# def add_args(parser, cfg, prefix=''):
#     for k, v in cfg.items():
#         if isinstance(v, str):
#             parser.add_argument('--' + prefix + k)
#         elif isinstance(v, int):
#             parser.add_argument('--' + prefix + k, type=int)
#         elif isinstance(v, float):
#             parser.add_argument('--' + prefix + k, type=float)
#         elif isinstance(v, bool):
#             parser.add_argument('--' + prefix + k, action='store_true')
#         elif isinstance(v, dict):
#             add_args(parser, v, k + '.')
#         elif isinstance(v, Iterable):
#             parser.add_argument('--' + prefix + k, type=type(v[0]), nargs='+')
#         else:
#             print('connot parse key {} of type {}'.format(prefix + k, type(v)))
#     return parser
#
#
# def check_file_exist(filename, msg_tmpl='file "{}" does not exist'):
#     if not osp.isfile(filename):
#         raise FileNotFoundError(msg_tmpl.format(filename))
#
#
# class Config(object):
#
#     @staticmethod
#     def fromfile(filename):
#         filename = osp.abspath(osp.expanduser(filename))
#         check_file_exist(filename)
#         if filename.endswith('.py'):
#             module_name = osp.basename(filename)[:-3]
#             if '.' in module_name:
#                 raise ValueError('Dots are not allowed in config file path.')
#             config_dir = osp.dirname(filename)
#             sys.path.insert(0, config_dir)
#             mod = import_module(module_name)
#             sys.path.pop(0)
#             cfg_dict = {
#                 name: value
#                 for name, value in mod.__dict__.items()
#                 if not name.startswith('__')
#             }
#         else:
#             raise IOError('Only py type are supported now!')
#         return Config(cfg_dict, filename=filename)
#
#     @staticmethod
#     def auto_argparser(description=None):
#         """Generate argparser from config file automatically (experimental)
#         """
#         partial_parser = ArgumentParser(description=description)
#         partial_parser.add_argument('config', help='config file path')
#         cfg_file = partial_parser.parse_known_args()[0].config
#         cfg = Config.fromfile(cfg_file)
#         parser = ArgumentParser(description=description)
#         parser.add_argument('config', help='config file path')
#         add_args(parser, cfg)
#         return parser, cfg
#
#     def __init__(self, cfg_dict=None, filename=None):
#         if cfg_dict is None:
#             cfg_dict = dict()
#         elif not isinstance(cfg_dict, dict):
#             raise TypeError('cfg_dict must be a dict, but got {}'.format(
#                 type(cfg_dict)))
#
#         super(Config, self).__setattr__('_cfg_dict', ConfigDict(cfg_dict))
#         super(Config, self).__setattr__('_filename', filename)
#         if filename:
#             with open(filename, 'r') as f:
#                 super(Config, self).__setattr__('_text', f.read())
#         else:
#             super(Config, self).__setattr__('_text', '')
#
#     @property
#     def filename(self):
#         return self._filename
#
#     @property
#     def text(self):
#         return self._text
#
#     def __repr__(self):
#         return 'Config (path: {}): {}'.format(self.filename,
#                                               self._cfg_dict.__repr__())
#
#     def __len__(self):
#         return len(self._cfg_dict)
#
#     def __getattr__(self, name):
#         return getattr(self._cfg_dict, name)
#
#     def __getitem__(self, name):
#         return self._cfg_dict.__getitem__(name)
#
#     def __setattr__(self, name, value):
#         if isinstance(value, dict):
#             value = ConfigDict(value)
#         self._cfg_dict.__setattr__(name, value)
#
#     def __setitem__(self, name, value):
#         if isinstance(value, dict):
#             value = ConfigDict(value)
#         self._cfg_dict.__setitem__(name, value)
#
#     def __iter__(self):
#         return iter(self._cfg_dict)

def file_to_config(file_name: str):

    if not file_name.endswith(".json"):
        raise TypeError(f"file_name must be json file, but file_name={file_name}")

    cfg = load_json(file_name)
    if "_base_" in cfg:
        base_files = cfg.pop("_base_")
        if isinstance(base_files, str):
            base_files = [base_files]

        assert isinstance(base_files, list), "_base_ must be a str or list"

        for base_file in base_files:
            cfg_dict, cfg_text = _file2dict(base_file)
            keys = cfg_dict.keys()
            for key in keys:
                if key.startswith('_') and key.endswith('_'):
                    update_value_of_dict(cfg_dict, key, cfg_dict[key])
            replace_kwargs_in_dict(cfg_dict)
            cfg = _merge_a_into_b(cfg, cfg_dict)

    return cfg


def get_base_config(cfg):
    if "_base_" in cfg:
        base_files = cfg.pop("_base_")
        if isinstance(base_files, str):
            base_files = [base_files]

        assert isinstance(base_files, list), "_base_ must be a str or list"

        for base_file in base_files:
            cfg_dict, cfg_text = _file2dict(base_file)
            keys = cfg_dict.keys()
            for key in keys:
                if key.startswith('_') and key.endswith('_'):
                    update_value_of_dict(cfg_dict, key, cfg_dict[key])
            replace_kwargs_in_dict(cfg_dict)
            cfg = _merge_a_into_b(cfg, cfg_dict)
    return cfg


def _file2dict(filename):
    BASE_KEY = '_base_'
    cfg_dict = load_json(filename)
    cfg_text = filename + '\n'
    with open(filename, 'r') as f:
        cfg_text += f.read()

    if BASE_KEY in cfg_dict:
        base_filename = cfg_dict.pop(BASE_KEY)
        base_filename = base_filename if isinstance(
            base_filename, list) else [base_filename]

        cfg_dict_list = list()
        cfg_text_list = list()
        for f in base_filename:
            _cfg_dict, _cfg_text = _file2dict(f)
            cfg_dict_list.append(_cfg_dict)
            cfg_text_list.append(_cfg_text)

        base_cfg_dict = dict()
        for _cfg_dict in cfg_dict_list:
            # base_cfg_dict.update(_cfg_dict)
            base_cfg_dict = _merge_a_into_b(base_cfg_dict, _cfg_dict)

        base_cfg_dict = _merge_a_into_b(cfg_dict, base_cfg_dict)
        cfg_dict = base_cfg_dict

        cfg_text_list.append(cfg_text)
        cfg_text = '\n'.join(cfg_text_list)

    return cfg_dict, cfg_text


def _merge_a_into_b(a, b):
    b = b.copy()
    for k, v in a.items():
        if not isinstance(v, dict):
            b[k] = v
        elif k not in b:
            b[k] = v
        else:
            if not isinstance(b[k], dict):
                b[k] = dict()   #
            b[k] = _merge_a_into_b(v, b[k])
    return b


def update_value_of_dict(_dict, old_value, new_value):
    if not isinstance(_dict, dict):
        return
    tmp = _dict
    for k, v in tmp.items():
        if isinstance(v, str) and v == old_value:
            _dict[k] = new_value
        else:
            if isinstance(v, dict):
                update_value_of_dict(_dict[k], old_value, new_value)
            elif isinstance(v, list):
                for _item in v:
                    update_value_of_dict(_item, old_value, new_value)


def replace_kwargs_in_dict(_dict):
    if not isinstance(_dict, dict):
        return
    _items = _dict.copy().items()
    for k, v in _items:
        if 'kwargs' == k:
            _kwargs = _dict.pop('kwargs')
            _dict.update(_kwargs)
        else:
            if isinstance(v, dict):
                replace_kwargs_in_dict(_dict[k])
            elif isinstance(v, list):
                for _item in v:
                    replace_kwargs_in_dict(_item)



if __name__ == '__main__':
    cfg = dict(a='1', b=2, kwargs=dict(c='ccc', b=[1, 2, 3]))
    cfg = Config(cfg)
    print(cfg.a)
    print(cfg.b)
    print(cfg.kwargs)
