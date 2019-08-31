from setuptools import setup, find_packages


setup(
    name="pypharma_nlp",
    #version='0.0.1',
    #url='https://github.com/examle.git', 
    #author='Author Name', 
    #author_email='author@gmail.com', 
    description='PyPharma NLP Package', 
    packages=find_packages(), 
    install_requires=[
        "biopython==1.74", 
        "matplotlib==3.1.1", 
        "ipykernel==5.1.2", 
        "seaborn==0.9.0", 
        "pandas==0.25.0", 
        "nltk==3.4.5", 
        "wget==3.2", 
        
        # For torch only
        "torch==1.2.0", 
        "boto3==1.9.215", 
        "tqdm==4.34.0", 
        "requests==2.22.0", 
        "regex==2019.8.19", 
        
        # For documentation
        "sphinx==2.2.0", 
        "sphinx-rtd-theme==0.4.3", 
    ]
)
