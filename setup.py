import os

import setuptools


setuptools.setup(
    name='MY_DEEP_LEARNING_LIB',
    version='0.0.1',
    keywords='demo',
    description='A package for deep learning.',
    long_description=open(
        os.path.join(
            os.path.dirname(__file__),
            'README.rst'
        )
    ).read(),
    author='wen',
    author_email='genning7@gmail.com',
    
    url='',
    download_url= 'https://github.com/Wen-Jian/ai-package/archive/v_0.0.1.tar.gz'
    packages=setuptools.find_packages(),
    license='MIT'
)