#!/usr/bin/env python3

from setuptools import setup
import torchcommon

if __name__ == '__main__':
    with open('README.md') as file:
        long_description = file.read()
    setup(
        name='torchcommon',
        packages=[
            'torchcommon',
            'torchcommon.nn',
            'torchcommon.optim',
        ],
        version='0.2',
        keywords=('pytorch', 'utilities'),
        description='Pytorch common utilities.',
        long_description_content_type='text/markdown',
        long_description=long_description,
        license='LGPL-2.1 License',
        author='xi',
        author_email='gylv@mail.ustc.edu.cn',
        url='https://github.com/XoriieInpottn/torchcommon',
        platforms='any',
        classifiers=[
            'Programming Language :: Python :: 3.6',
            'Programming Language :: Python :: 3.7',
            'Programming Language :: Python :: 3.8',
        ],
        include_package_data=True,
        zip_safe=True,
        install_requires=[
            'torch',
            'numpy'
        ]
    )
