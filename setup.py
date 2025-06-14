from setuptools import setup, find_packages
from typing import List
def get_requirements(file_path: str) -> List[str]:
    """
    This function reads a requirements file and returns a list of packages.
    It removes any '-e .' entry which is used for editable installs.
    """
    requirements=[]
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace('\n', '') for req in requirements]
        if '-e .' in requirements:
            requirements.remove('-e .')

setup(
    name='MLproject',
    version='0.1',
    author='Sourav',
    author_email='chouhansourav4@gmail.com',
    packages=find_packages(),
    install_requires= get_requirements('requirements.txt'),
)