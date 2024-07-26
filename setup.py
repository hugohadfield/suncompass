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
    package_data={'': ['3_224_224_resnet_baseline_5_all.pth']},
    include_package_data=True,
    packages=find_packages(),
    install_requires=[
        'torch',
        'torchvision',
        'pandas',
        'numpy',
        'matplotlib',
        'tqdm',
        'click',
        'Pillow',
        'tensorboard',
        'suncalc'
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