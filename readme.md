Warning: broken

ML in Rust is a waste of time
Python is much better

## troubleshooting

unaligned loading is ok
tokenizer is ok
normalized() is ok

first layer_norm has tiny error

small errors after each ln layer

probs is all wrong

todo
- check time mixing with fake data [1,2,3,4] [5,6,7,8] [9,10,11,12]...
- check space mixing

## Usage

1. get model params
First, dump the model with inspectxxxx.py
TK

token

20B_tokenizer.json


Then,

```rs
TK
```

## To do

- replace assert with error
- remove device jankyness
