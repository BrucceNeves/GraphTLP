from setuptools import setup, find_packages

VERSION = '0.0.1' 
DESCRIPTION = 'Transductive Label Propagation'
LONG_DESCRIPTION = 'Transductive label propagation classic algorithms'

setup(
        name="transductive_label_propagation", 
        version=VERSION,
        author="Brucce",
        author_email="<youremail@email.com>",
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        packages=find_packages(),
        install_requires=['pandas', 'numpy', 'networkx'],
        keywords=['network', 'transductive label propagation']
)
