Warning: experimental

dfdx cannot yet utilize 100% of your GPU. As a result, this is slower than ggml or numba/numpy.

## Usage

1. get model


```shell
wget2 https://huggingface.co/BlinkDL/rwkv-4-pile-430m/resolve/main/RWKV-4-Pile-430M-20220808-8066.pth

# get convert.py from https://github.com/iacore/rwkv-np
python convert.py RWKV-4-Pile-430M-20220808-8066.pth
```

2. infer

```shell
cargo run --example infer --release
```

***

In theory, using bigger RWKV is also possible if you got enough memory. Just remember to change the Rust model type and model path in `examples/infer.rs`.
