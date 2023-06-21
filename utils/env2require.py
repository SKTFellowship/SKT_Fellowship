#!pip install ruamel.yaml

'''
This code is only for pip
In case of dependencies
You need to download it directly from the terminal

In case of pytorch
Use Public documents https://pytorch.org/
'''

import ruamel.yaml

yaml = ruamel.yaml.YAML()
data = yaml.load(open('environment.yaml'))

requirements = []
for pip in data['dependencies'][-1]["pip"]:
    # if isinstance(pip, str):
    #     print(pip)
    #     package, package_version, python_version = pip.split('=')
    #     if python_version == '0':
    #         continue
    #     requirements.append(package + '==' + package_version)
    # elif isinstance(pip, dict):
    #     for preq in pip.get('pip', []):
    #         requirements.append(preq)
    requirements.append(pip)

with open('requirements.txt', 'w') as fp:
    for requirement in requirements:
        print(requirement, file=fp)