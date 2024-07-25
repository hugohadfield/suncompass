from setuptools import setup, find_packages

setup(
    name='suncompass',
    version='0.0.1',
    description='Code for estimating the direction of the sun from a single outdoor image.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Hugo Hadfield',
    author_email='hadfield.hugo@gmail.com',
    url='https://github.com/hugohadfield/suncompass',
    package_data={'': ['suncompass/3_224_224_resnet_baseline_5_all.pth']},
    include_package_data=True,
    packages=find_packages(),
    install_requires=[
        'torch>=1.7.0',
        'torchvision>=0.8.0',
        'pandas>=1.1.0',
        'numpy>=1.18.0',
        'matplotlib>=3.2.0',
        'tqdm>=4.48.0',
        'click>=7.1.0',
        'Pillow>=7.2.0',
        'tensorboard>=2.3.0',
        'suncalc>=0.2.0'
    ],
    entry_points={
        'console_scripts': [
            'add_sun_vector=add_sun_vector:main',
            'extract_motion=extract_motion:main',
            'pretrained_resnet=pretrained_resnet:main',
            'run_model_on_images=run_model_on_images:main'
        ]
    },
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
    python_requires='>=3.7',
)