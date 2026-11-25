import os
from setuptools import setup

package_name = 'ur5_mujoco'


def recursive_data_files(src_root, dest_root):
    """Install a directory tree (needed for the menagerie meshes)."""
    out = []
    for dirpath, _, filenames in os.walk(src_root):
        if not filenames:
            continue
        dest = os.path.join(dest_root, os.path.relpath(dirpath, src_root))
        out.append((dest, [os.path.join(dirpath, f) for f in filenames]))
    return out


data_files = [
    ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
    ('share/' + package_name, ['package.xml']),
]
data_files += recursive_data_files('models', os.path.join('share', package_name, 'models'))

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=data_files,
    install_requires=['setuptools'],
    zip_safe=True,
    description='MuJoCo UR5e simulator: scenarios and robot models',
    license='Apache-2.0',
)
