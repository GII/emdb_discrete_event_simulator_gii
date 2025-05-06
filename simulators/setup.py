from setuptools import find_packages, setup

package_name = 'simulators'

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
    maintainer='sergio',
    maintainer_email='sergio.martinez3@udc.es',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'simulator_discrete = simulators.sim_discrete:main',
            'fruit_shop_simulator = simulators.fruit_shop_sim_discrete:main', 
            'pump_panel_simulator = simulators.pump_panel_sim_discrete:main'
        ],
    },
)
