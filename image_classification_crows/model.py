# AUTOGENERATED! DO NOT EDIT! File to edit: ../publish_model.ipynb.

# %% auto 0
__all__ = ['learn_exported', 'categories', 'classify_image', 'hello_world']

# %% ../publish_model.ipynb 3
from fastbook import *
from nbdev import *
from fastai.vision.widgets import *
from duckduckgo_search import ddg_images

doc(print)

# %% ../publish_model.ipynb 19
learn_exported=load_learner(Path()/'export.pkl')
categories='crow', 'raven'

# %% ../publish_model.ipynb 21
def classify_image(img):
  pred, idx, probs = learn_exported.predict(img)
  return dict(zip(categories, map(float, probs)))


# %% ../publish_model.ipynb 25
def hello_world(name): 
    return f"🌏 Hello {name} 🌎"
