#!/usr/bin/env python3

import io

__all__ = [
    'Option',
    'get_options',
    'BaseConfig'
]


class Option(object):

    def __init__(self, value, type=None, help=None):
        self.value = value
        self.type = type
        self.help = help
        if self.type is None and self.value is not None:
            self.type = self.value.__class__


def get_options(obj):
    options = {}
    class_list = []
    base = obj if isinstance(obj, type) else obj.__class__
    while base is not BaseConfig:
        class_list.append(base)
        base = base.__base__
    for clazz in reversed(class_list):
        for name, value in clazz.__dict__.items():
            if not name.startswith('_'):
                options[name] = value
    return options


class BaseConfig(object):

    def __init__(self, args=None):
        for name, value in get_options(self).items():
            if isinstance(value, Option):
                value = value.value
            setattr(self, name, value)

        if args is not None:
            self.load(args)

    def load(self, args):
        for name, value in self.__dict__.items():
            if name.startswith('_'):
                continue
            if hasattr(args, name):
                setattr(self, name, getattr(args, name))

    def __str__(self):
        pairs = []
        max_len = 0
        for name, value in self.__dict__.items():
            if name.startswith('_'):
                continue

            if value is None:
                value = 'None'
            elif isinstance(value, str):
                value = f'\'{value}\''
            else:
                value = str(value)

            break_line = value.find('\n')
            if break_line >= 0:
                value = value[:break_line] + '...'

            pairs.append((name, value))

            name_len = len(name)
            if name_len > max_len:
                max_len = name_len

        buffer = io.StringIO()
        for name, value in pairs:
            buffer.write(name.rjust(max_len))
            buffer.write(': ')
            buffer.write(value)
            buffer.write('\n')
        return buffer.getvalue()
