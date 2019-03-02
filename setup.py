from os import path
from setuptools import setup, find_packages


install_requires = ['numpy>=1.12.0']

about = {}
here = path.abspath(path.dirname(__file__))
with open(path.join(here, 'dnnet', '__version__.py'), 'r', encoding='utf-8') as f:
    exec(f.read(), about)

with open('README.md', 'r', encoding='utf-8') as f:
    readme = f.read()

setup(
    name=about['__title__'],
    version=about['__version__'],
    description=about['__description__'],
    long_description=readme,
    long_description_content_type='text/markdown',
    author=about['__author__'],
    author_email=about['__author_email__'],
    url=about['__url__'],
    packages=find_packages(),
    include_package_data=True,
    python_requires='>=3.4',
    install_requires=install_requires,
    license=about['__license__'],
    zip_safe=False,
    classifiers = [
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Natural Language :: English',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Topic :: Scientific/Engineering :: Artificial Intelligence'
    ],
)
