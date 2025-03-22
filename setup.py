from setuptools import setup, find_packages

setup(
    name='screenshot-ocr',
    version='0.2.9',
    packages=find_packages(),  # Automatically find packages (if you have them)
    py_modules=['screenshot_ocr', 'ocr_base', 'ollama_engine'],  
    install_requires=[
        'PyYAML',
        'httpx',
        'prompt_toolkit',
        'Pillow'
    ],
    entry_points={
        'console_scripts': [
            'screenshot-ocr=screenshot_ocr:real_main',  # Creates an executable named 'screenshot-ocr'
        ],
    },
)
