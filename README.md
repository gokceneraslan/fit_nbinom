# fit_nbinom
Negative binomial maximum likelihood estimate implementation in Python using L-BFGS-B

# installation

```python
python setup.py install
```

# usage

## 1. Fit negative-binomial model and show summary

```python
from fit_nbinom import fit_nbinom
import numpy as np

# X is a list or a numpy array representing the data
X = np.array([16, 18, 11, 19, 20, 3, 2, 11, 8, 5])

# 
res = fit_nbinom().fit(X=X)

res.summary()
```

output:
```
parameter      estimate    std err    95%CI lower    95%CI upper
-----------  ----------  ---------  -------------  -------------
size            3.19747   0.584826        2.05123        4.34371
mu             11.3       0.715784        9.89709       12.7029
```

`tablefmt` argument defines format of summary table.
Available format is the same as [python-tabulate](https://bitbucket.org/astanin/python-tabulate/src/master/).

example: `latex` format
```python
res.summary(tablefmt="latex")
```

output:
```
\begin{tabular}{lrrrr}
\hline
 parameter   &   estimate &   std err &   95\%CI lower &   95\%CI upper \\
\hline
 size        &    3.19747 &  0.584678 &       2.05153 &       4.34342 \\
 mu          &   11.3     &  0.715784 &       9.89709 &      12.7029  \\
\hline
\end{tabular}
```

## 2. Get estimates of parameters
```python
parameters = res.params()
print(parameters)
```

output:
```
{'size': 3.1974721271555056, 'mu': 11.300000225025348}
```

## 3. Get standard errors of parameters
```python
parameters = res.params()
print(parameters)
```

output:
```
{'size': 0.5846775256549968, 'mu': 0.7157840010743496}
```

# description of output parameters
In the probability mass function of negative binomial distribution below:
- `size` parameter refers to <img src="https://latex.codecogs.com/gif.latex?\large&space;r" />
- `mu` parameter refers to <img src="https://latex.codecogs.com/gif.latex?\large&space;\mu" />

<img src="https://latex.codecogs.com/gif.latex?\large&space;\P&space;(X=x)&space;=&space;\binom{x&plus;r-1}{x}\left(&space;\frac{r}{r&plus;\mu}&space;\right)^r&space;\left(&space;\frac{\mu}{r&plus;\mu}&space;\right)^x" />