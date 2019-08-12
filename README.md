# Using the notebooks with jupyter and virtualenv

Create a virtual environment

```
virtualenv -p python3 virtualenv
source venv/bin/activate
pip install -r requirements
```

If desired, you can create a jupyter kernel for the virtual environment

```
pip install jupyter ipykernel
ipython kernel install --user --name=pypharma\_nlp
```

And then start the jupyter

```
jupyter notebook
```
