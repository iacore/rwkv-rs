#![allow(non_camel_case_types)]

use std::path::Path;

use dfdx::prelude::*;

use ::safetensors::SafeTensors;
use memmap2::MmapOptions;
use Tensor1D as V;
use Tensor2D as M;

fn mix<const N: usize>(mix: V<N>, p0: V<N>, p1: V<N>) -> V<N> {
    p1 * mix.clone() + p0 * (-mix + 1.)
}

#[derive(Clone)]
pub struct ATT<const N: usize> {
    pub decay: V<N>,   // time_decay
    pub bonus: V<N>,   // time_first
    pub mix_k: V<N>,   // time_mix_k
    pub mix_v: V<N>,   // time_mix_v
    pub mix_r: V<N>,   // time_mix_r
    pub wk: M<N, N>,   // key.weight
    pub wv: M<N, N>,   // value.weight
    pub wr: M<N, N>,   // receptance.weight
    pub wout: M<N, N>, // output.weight
}

#[derive(Clone)]
pub struct ATTState<const N: usize> {
    pub x: V<N>,
    pub num: V<N>,
    pub den: V<N>,
}

impl<const N: usize> ATTState<N> {
    pub fn zeros(dev: &Cpu) -> Self {
        Self {
            x: dev.zeros(),
            num: dev.zeros(),
            den: dev.zeros(),
        }
    }
}

impl<const N: usize> ATT<N> {
    pub fn zeros(dev: &Cpu) -> Self {
        Self {
            decay: dev.zeros(),
            bonus: dev.zeros(),
            mix_k: dev.zeros(),
            mix_v: dev.zeros(),
            mix_r: dev.zeros(),
            wk: dev.zeros(),
            wv: dev.zeros(),
            wr: dev.zeros(),
            wout: dev.zeros(),
        }
    }

    pub fn forward(&self, x: V<N>, last: ATTState<N>) -> (V<N>, ATTState<N>) {
        let decay = self.decay.clone();
        let bonus = self.bonus.clone();
        let mix_k = self.mix_k.clone();
        let mix_v = self.mix_v.clone();
        let mix_r = self.mix_r.clone();
        let wk = self.wk.clone();
        let wv = self.wv.clone();
        let wr = self.wr.clone();
        let wout = self.wout.clone();

        let k = matmul(wk, mix(mix_k, last.x.clone(), x.clone()));
        let v = matmul(wv, mix(mix_v, last.x.clone(), x.clone()));
        let r = matmul(wr, mix(mix_r, last.x, x.clone()));

        let _0 = exp(bonus + k.clone());

        let wkv = (last.num.clone() + _0.clone() * v.clone()) / (last.den.clone() + _0);
        let rwkv = sigmoid(r) * wkv;

        let _1 = exp(-exp(decay));
        let _2 = exp(k);
        let num = _1.clone() * last.num + _2.clone() * v;
        let den = _1 * last.den + _2;

        (matmul(wout, rwkv), ATTState { x, num, den })
    }
}

#[derive(Clone)]
pub struct FFN<const N: usize, const N1: usize> {
    pub mix_k: V<N>,  // time_mix_k
    pub mix_r: V<N>,  // time_mix_r
    pub wk: M<N1, N>, // key.weight
    pub wv: M<N, N1>, // value.weight
    pub wr: M<N, N>,  // receptance.weight
}

impl<const N: usize, const N1: usize> FFN<N, N1> {
    pub fn zeros(dev: &Cpu) -> Self {
        Self {
            mix_k: dev.zeros(),
            mix_r: dev.zeros(),
            wk: dev.zeros(),
            wv: dev.zeros(),
            wr: dev.zeros(),
        }
    }
    pub fn forward(&self, dev: &Cpu, x: V<N>, last_x: V<N>) -> (V<N>, V<N>) {
        let mix_k = self.mix_k.clone();
        let mix_r = self.mix_r.clone();
        let wk = self.wk.clone();
        let wv = self.wv.clone();
        let wr = self.wr.clone();

        let k = matmul(wk, mix(mix_k, last_x.clone(), x.clone()));
        let r = matmul(wr, mix(mix_r, last_x, x.clone()));
        let vk = matmul(wv, maximum(k, dev.zeros()).square());

        (sigmoid(r) * vk, x)
    }
}

#[derive(Clone)]
pub struct LN<const N: usize> {
    pub weight: V<N>,
    pub bias: V<N>,
}

impl<const N: usize> LN<N> {
    pub fn zeros(dev: &Cpu) -> Self {
        Self {
            weight: dev.zeros(),
            bias: dev.zeros(),
        }
    }

    pub fn layer_norm(&self, x: V<N>) -> V<N> {
        let w = self.weight.clone();
        let b = self.bias.clone();
        x.normalize(0.) * w + b
    }
}

#[derive(Clone)]
pub struct RWKVBlock<const N_EMBED: usize, const N_EMBED_TIMES_4: usize> {
    pub ln1: LN<N_EMBED>,
    pub att: ATT<N_EMBED>,
    pub ln2: LN<N_EMBED>,
    pub ffn: FFN<N_EMBED, N_EMBED_TIMES_4>,
}

impl<const N_EMBED: usize, const N_EMBED_TIMES_4: usize> RWKVBlock<N_EMBED, N_EMBED_TIMES_4> {
    pub fn zeros(dev: &Cpu) -> Self {
        Self {
            ln1: LN::zeros(dev),
            att: ATT::zeros(dev),
            ln2: LN::zeros(dev),
            ffn: FFN::zeros(dev),
        }
    }
}

#[derive(Clone)]
pub struct RWKVState<const N: usize> {
    pub att_state: ATTState<N>,
    pub ffn_state: V<N>,
}

impl<const N: usize> RWKVState<N> {
    pub fn zeros(dev: &Cpu) -> Self {
        Self {
            att_state: ATTState::zeros(dev),
            ffn_state: dev.zeros(),
        }
    }
}

#[derive(Clone)]
pub struct RWKV<
    const N_LAYER: usize,
    const N_EMBED: usize,
    const N_EMBED_TIMES_4: usize,
    const N_VOCAB: usize,
> {
    pub emb: Tensor2D<N_VOCAB, N_EMBED>, // emb.weight
    pub ln_in: LN<N_EMBED>,              // blocks.0.ln0
    pub blocks: [RWKVBlock<N_EMBED, N_EMBED_TIMES_4>; N_LAYER],
    pub ln_out: LN<N_EMBED>,              // ln_out
    pub head: Tensor2D<N_VOCAB, N_EMBED>, // head.weight
}

impl<
        const N_LAYER: usize,
        const N_EMBED: usize,
        const N_EMBED_TIMES_4: usize,
        const N_VOCAB: usize,
    > RWKV<N_LAYER, N_EMBED, N_EMBED_TIMES_4, N_VOCAB>
{
    pub fn zeros(dev: &Cpu) -> Self {
        Self {
            emb: dev.zeros(),
            ln_in: LN::zeros(dev),
            blocks: std::array::from_fn(|_| RWKVBlock::zeros(dev)),
            ln_out: LN::zeros(dev),
            head: dev.zeros(),
        }
    }

    pub fn load_safetensors<P: AsRef<Path>>(
        &mut self,
        path: P,
    ) -> Result<(), dfdx::tensor::safetensors::Error> {
        let f = std::fs::File::open(path)?;
        let buffer = unsafe { MmapOptions::new().map(&f)? };
        let xs = SafeTensors::deserialize(&buffer)?;

        let Self {
            emb,
            ln_in,
            blocks,
            ln_out,
            head,
        } = self;
        _cp(&xs, "emb.weight", emb)?;
        _cp(&xs, "head.weight", head)?;
        _cp_ln(&xs, "blocks.0.ln0", ln_in)?;
        _cp_ln(&xs, "ln_out", ln_out)?;
        for (i, block) in blocks.into_iter().enumerate() {
            _cp_block(&xs, &format!("blocks.{i}"), block)?;
        }

        Ok(())
    }

    pub fn forward(
        &self,
        dev: &Cpu,
        token: usize,
        state: RWKVState<N_EMBED>,
    ) -> (V<N_VOCAB>, RWKVState<N_EMBED>) {
        let x = self.emb.clone().select(dev.tensor(token));
        let x = self.ln_in.layer_norm(x);

        let (x, state) = self.blocks.iter().fold(
            (x, state),
            |(
                x,
                RWKVState {
                    att_state,
                    ffn_state,
                },
            ),
             block| {
                let x_ = block.ln1.layer_norm(x.clone());
                let (dx, att_state) = block.att.forward(x_, att_state);
                let x = x + dx;

                let x_ = block.ln2.layer_norm(x.clone());
                let (dx, ffn_state) = block.ffn.forward(dev, x_, ffn_state);
                let x = x + dx;

                (
                    x,
                    RWKVState {
                        att_state,
                        ffn_state,
                    },
                )
            },
        );

        let x = self.ln_out.layer_norm(x);
        let x = matmul(self.head.clone(), x); // "attention" head

        (softmax(x), state)
    }
}

fn matmul<const N0: usize, const N1: usize>(lns: M<N0, N1>, rhs: V<N1>) -> V<N0> {
    let rhs = rhs.reshape::<Rank2<N1, 1>>();
    let res: M<N0, 1> = lns.matmul(rhs);
    res.reshape()
}

fn _cp_block<const N_EMBED: usize, const N_EMBED_TIMES_4: usize>(
    xs: &SafeTensors,
    prefix: &str,
    block: &mut RWKVBlock<N_EMBED, N_EMBED_TIMES_4>,
) -> Result<(), dfdx::tensor::safetensors::Error> {
    let RWKVBlock { ln1, att, ln2, ffn } = block;

    _cp_ln(xs, &format!("{prefix}.ln1"), ln1)?;
    _cp_ln(xs, &format!("{prefix}.ln2"), ln2)?;

    {
        let ATT {
            decay,
            bonus,
            mix_k,
            mix_v,
            mix_r,
            wk,
            wv,
            wr,
            wout,
        } = att;
        _cp(xs, &format!("{prefix}.att.time_decay"), decay)?;
        _cp(xs, &format!("{prefix}.att.time_first"), bonus)?;
        _cp(xs, &format!("{prefix}.att.time_mix_k"), mix_k)?;
        _cp(xs, &format!("{prefix}.att.time_mix_v"), mix_v)?;
        _cp(xs, &format!("{prefix}.att.time_mix_r"), mix_r)?;
        _cp(xs, &format!("{prefix}.att.key.weight"), wk)?;
        _cp(xs, &format!("{prefix}.att.value.weight"), wv)?;
        _cp(xs, &format!("{prefix}.att.receptance.weight"), wr)?;
        _cp(xs, &format!("{prefix}.att.output.weight"), wout)?;
    }

    {
        let FFN {
            mix_k,
            mix_r,
            wk,
            wv,
            wr,
        } = ffn;
        _cp(xs, &format!("{prefix}.ffn.time_mix_k"), mix_k)?;
        _cp(xs, &format!("{prefix}.ffn.time_mix_r"), mix_r)?;
        _cp(xs, &format!("{prefix}.ffn.key.weight"), wk)?;
        _cp(xs, &format!("{prefix}.ffn.value.weight"), wv)?;
        _cp(xs, &format!("{prefix}.ffn.receptance.weight"), wr)?;
    }

    Ok(())
}

fn _cp_ln<const N: usize>(
    xs: &SafeTensors,
    prefix: &str,
    ln_in: &mut LN<N>,
) -> Result<(), dfdx::tensor::safetensors::Error> {
    _cp(xs, &format!("{prefix}.weight"), &mut ln_in.weight)?;
    _cp(xs, &format!("{prefix}.bias"), &mut ln_in.bias)?;
    Ok(())
}

/// copy tensor
fn _cp<S: Shape, D: CopySlice<f32>, T>(
    tensors: &SafeTensors,
    tensor_name: &str,
    tensor: &mut Tensor<S, f32, D, T>,
) -> Result<(), dfdx::tensor::safetensors::Error> {
    let tensor_view = tensors.tensor(tensor_name)?;

    let shape_expected: Vec<usize> = tensor_view
        .shape()
        .into_iter()
        .cloned()
        .filter(|x| *x != 1)
        .collect();

    let shape_actual = &tensor.shape();
    for i in 0..S::NUM_DIMS {
        assert_eq!(shape_actual.concrete()[i], shape_expected[i]);
    }

    let data = tensor_view.data();
    tensor.copy_from(unsafe { align_to_f32(data) });
    Ok(())
}

/// data may be unaligned
unsafe fn align_to_f32(data: &[u8]) -> &[f32] {
    std::slice::from_raw_parts(data.as_ptr() as *const f32, data.len() / 4)
}

pub type RWKV_430m = RWKV<24, 1024, 4096, 50277>;

#[test]
fn test_load_model() -> Result<(), dfdx::tensor::safetensors::Error> {
    let dev: Cpu = Default::default();
    let mut model = RWKV_430m::zeros(&dev);
    model.load_safetensors("RWKV-4-Pile-430M-20220808-8066.safetensors")?;
    Ok(())
}
