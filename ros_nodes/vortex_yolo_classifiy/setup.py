from setuptools import setup

package_name = 'vortex_yolo_classifiy'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/classifier_node.launch.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='gard',
    maintainer_email='gard@example.com',
    description='PyTorch image classifier node',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'classifier_node = vortex_yolo_classifiy.classifier_node:main',
        ],
    },
)