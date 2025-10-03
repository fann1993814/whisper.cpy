import sys

from os.path import abspath, join, dirname
from setuptools import find_packages, setup

this_dir = abspath(dirname(__file__))
# In Python3 TypeError: a bytes-like object is required, not 'str'
if sys.version_info[0] < 3:
    long_description = 'Python wrapper for Whisper.cpp'
else:
    with open(join(this_dir, 'README.md'), encoding='utf-8') as file:
        long_description = file.read()

setup(
    name='whispercpy',
    version='0.2.1',
    description='Python wrapper for Whisper.cpp',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/fann1993814/whisper.cpy',
    author='Jason Fan',
    author_email='fann1993814@gmail.com',
    license='MIT',
    classifiers=[
        'Intended Audience :: Developers',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Topic :: Multimedia',
        'Topic :: Multimedia :: Sound/Audio',
        'Topic :: Multimedia :: Sound/Audio :: Speech'
    ],
    keywords='speech recognition,speech to text,STT,ASR,NLP',
    python_requires='>=3.8',
    packages=find_packages(exclude=['examples*']),
    package_dir={'whispercpy': 'whispercpy'},
    package_data={'': ['*.*', 'data/*']},
    include_package_data=True,
    install_requires=['numpy>=1.25', 'scipy>=1.13.0', 'sounddevice>=0.5.0']
)
