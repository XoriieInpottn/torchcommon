#!/usr/bin/env python3

import io

__all__ = [
    'BaseConfig'
]


class BaseConfig(object):

    def __init__(self, args=None):
        if args is not None:
            self.load(args)

    def load(self, args):
        for name, value in self._get_attributes().items():
            if name.startswith('_'):
                continue
            if hasattr(args, name):
                setattr(self, name, getattr(args, name))

    def __str__(self):
        pairs = []
        max_len = 0
        for name, value in self._get_attributes().items():
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
                value = value[:break_line]

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

    def _get_attributes(self):
        d = {}
        for name, value in self.__class__.__dict__.items():
            d[name] = value
        for name, value in self.__dict__.items():
            d[name] = value
        return d
