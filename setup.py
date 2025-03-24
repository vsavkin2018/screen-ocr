from setuptools import setup, find_packages

setup(
    name='screenshot-ocr',
    version='0.3.0',
    packages=find_packages(),  # Automatically find packages (if you have them)
    py_modules=['screenshot_ocr', 'ocr_base', 
	'ollama_engine', 'tesseract_engine'],  
    install_requires=[
        'PyYAML',
        'httpx',
        'prompt_toolkit',
        'Pillow',
	'pytesseract'
    ],
    entry_points={
        'console_scripts': [
            'screenshot-ocr=screenshot_ocr:real_main',  # Creates an executable named 'screenshot-ocr'
        ],
    },
)
