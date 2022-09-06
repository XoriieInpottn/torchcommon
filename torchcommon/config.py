#!/usr/bin/env python3

import io

__all__ = [
    'BaseConfig'
]


class BaseConfig(object):

    def load(self, args):
        for name, value in self.__dict__.items():
            if name.startswith('_') or callable(value):
                continue
            if hasattr(args, name):
                setattr(self, name, getattr(args, name))

    def __str__(self):
        pairs = []
        max_len = 0
        for name, value in self.__dict__.items():
            if name.startswith('_') or callable(value):
                continue
            if value is None:
                value = 'None'
            elif isinstance(value, str):
                value = f'\'{value}\''
            else:
                value = str(value)
            l = len(name)
            if l > max_len:
                max_len = l
            pairs.append((name, value))

        buffer = io.StringIO()
        for name, value in pairs:
            buffer.write(name.rjust(max_len))
            buffer.write(': ')
            buffer.write(value)
            buffer.write('\n')
        return buffer.getvalue()
