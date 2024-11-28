import inspect


class Registry:
    """
    注册器，用于注册类和函数
    """
    def __init__(self, name):
        self._name = name
        self._module_dict = dict()

    def __len__(self):
        return len(self._module_dict)

    def __contains__(self, key):
        return self.get(key) is not None

    @property
    def name(self):
        return self._name

    @property
    def module_dict(self):
        return self._module_dict

    def get(self, key):
        return self._module_dict.get(key, None)

    def register_module(self, name=None,
                        force: bool=False,
                        module=None):

        if module is not None:
            self._register_module(module, module_name=name, force=force)
            return module

        if not (name is None or isinstance(name, str)):
            raise TypeError(f'name must be a str, but got {type(name)}')

        def _register(m):
            self._register_module(m, module_name=name, force=force)
            return m
        return _register

    def _register_module(self, module, module_name, force=False):

        if inspect.isclass(module) or callable(module):
            if module_name is None:
                module_name = module.__name__
            if not force and module_name in self._module_dict:
                raise KeyError(f'{module_name} is already registered '
                               f'in {self.name}')
            self._module_dict[module_name] = module
        else:
            raise TypeError('module must be a class or function'
                            f'but got {type(module)}')


def build_from_cfg(cfg: dict,
                   registry: Registry,
                   update_args: dict = None):
    """
    通过配置文件创建一个类，或者执行一个函数
    """
    args = cfg.copy()
    module_type = args.pop('type')
    if isinstance(module_type, str):
        module_obj = registry.get(module_type)
        if module_obj is None:
            raise KeyError(f'{module_type} is not in the {registry.name} registry')

        if update_args is not None:
            args.update(**update_args)
        return module_obj(**args)
    else:
        raise TypeError(
            f'type must be a str, but got {type(module_type)}')


if __name__=="__main__":

    UPLAOD = Registry("upload")

    class BaseUPLoad:
        def __init__(self, txt=""):
            self.txt = txt

        def __call__(self, ):
            print(self.__class__.__name__, ":", self.txt)


    @UPLAOD.register_module()
    class UpLoadA(BaseUPLoad):
        def __init__(self, **kwargs):
            super(UpLoadA, self).__init__(**kwargs)

    @UPLAOD.register_module()
    class UpLoadB(BaseUPLoad):
        def __init__(self, **kwargs):
            super(UpLoadB, self).__init__(**kwargs)

    cfg = dict(type="UpLoadA", txt="hello")
    op1 = build_from_cfg(cfg, UPLAOD)
    op1()

    cfg = dict(type="UpLoadB", txt="world")
    op2 = build_from_cfg(cfg, UPLAOD)
    op2()
