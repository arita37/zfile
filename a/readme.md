

###Only in linux


conda create -n test python=3.7

conda install *.tar.bz2

conda update --all

pip install ipython  ipykernel


#### you can just ipython




Some toy data matrices are available in the [`tst-data`](https://github.com/amzn/pecos/tree/mainline/test/tst-data/xmc/xlinear) folder. 

PECOS constructs a hierarchical label tree and learns linear models recursively (e.g., XR-Linear):
```python
>>> from pecos.xmc.xlinear.model import XLinearModel
>>> from pecos.xmc import Indexer, LabelEmbeddingFactory

# Build hierarchical label tree and train a XR-Linear model
>>> label_feat = LabelEmbeddingFactory.create(Y, X)
>>> cluster_chain = Indexer.gen(label_feat)
>>> model = XLinearModel.train(X, Y, C=cluster_chain)
>>> model.save("./save-models")
```

After learning the model, we do prediction and evaluation 
```python
>>> from pecos.utils import smat_util
>>> Yt_pred = model.predict(Xt)
# print precision and recall at k=10
>>> print(smat_util.Metrics.generate(Yt, Yt_pred))
```

PECOS also offers optimized C++ implementation for fast real-time inference
```python
>>> model = XLinearModel.load("./save-models", is_predict_only=True)
>>> for i in range(X_tst.shape[0]):
>>>   y_tst_pred = model.predict(X_tst[i], threads=1)
```



