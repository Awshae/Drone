from setuptools import find_packages, setup

package_name = 'firefighting_pkg'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='fariza',
    maintainer_email='ri06nuha@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'oakd_yolo = firefighting_pkg.yolo_node:main',
            'controller = firefighting_pkg.controller:main',
            'firebase_bridge = firefighting_pkg.firebase_bridge_node:main',
            'pre_flight = firefighting_pkg:pre_flight_check:main'
        ],
    },
)
