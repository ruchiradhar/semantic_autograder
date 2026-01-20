# setup.py is essential to use project as a package and deploy in pypi
# setuptools is used to import two important libaries
from setuptools import find_packages,setup #find packages available in the project
from typing import List # List is used in the requirements function as a part of setup()

# find_packages looks for __init__.py in directory and considers it as a package which can be imported
# setup is used to provide metadata information for the whole project
# for setup, first create requirement fetching function

HYPHEN_E_DOT='-e .'

def get_requirements(file_path:str)-> List[str]:
    ''' 
    this function will return list of requirements
    '''
    requirements=[]
    with open (file_path) as file_obj:
        requirements=file_obj.readlines()
        requirements= [req.replace('\n','') for req in requirements]

        if HYPHEN_E_DOT in requirements: #used to prevent e dot getting added to list of requirements
            requirements.remove(HYPHEN_E_DOT)


setup (
    name='semantic_autograder',
    version='0.1.0',
    author='Ruchira',
    author_email='rudh@di.ku.dk',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt'),
    entry_points={
        'console_scripts': [
            'sa-train=src.pipeline.train_pipeline:main',
            'sa-predict=src.pipeline.predict_pipeline:cli',
        ]
    }
)