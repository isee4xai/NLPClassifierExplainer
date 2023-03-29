
# Always prefer setuptools over distutils
from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()

setup(
  
    name="NLPClassifierExplainer",  # Required
    
    version="0.0.2",  # Required
    url="https://github.com/isee4xai/NLPClassifierExplainer",  # Optional 
    author='b fleisch',
    author_email='bruno.fleisch at bt.com',
    
    # When your source code is in a subdirectory under the project root, e.g.
    # `src/`, it is necessary to specify the `package_dir` argument.
    package_dir={"": "src"},  # Optional
    # You can just specify package directories manually here if your project is
    # simple. Or you can use find_packages().
    #
    # Alternatively, if you just want to distribute a single Python file, use
    # the `py_modules` argument instead as follows, which will expect a file
    # called `my_module.py` to exist:
    #
    #   py_modules=["my_module"],
    #
    packages=find_packages(where="src"),  # Required
    python_requires=">=3.3, <4",
    
    # This field lists other packages that your project depends on to run.
    # Any package you put here will be installed by pip when your project is
    # installed, so they must be valid existing projects.
    #
    # For an analysis of "install_requires" vs pip's requirements files see:
    # https://packaging.python.org/discussions/install-requires-vs-requirements/
    
    install_requires=["nltk", "scikit-learn","joblib", "numpy", "stop-words","wordcloud","gensim"],  # Optional
    
  
    project_urls={  # Optional
        "Bug Reports": "https://github.com/isee4xai/NLPClassifierExplainer/issues",
        "Source": "https://github.com/isee4xai/NLPClassifierExplainer/",
    },
)
