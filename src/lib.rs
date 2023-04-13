#![allow(non_camel_case_types)]

use std::path::Path;

use dfdx::prelude::*;

use ::safetensors::SafeTensors;
use memmap2::MmapOptions;
use rand::distributions::Uniform;
use tracing::trace;
use Tensor1D as V;
use Tensor2D as M;

fn mix<const N: usize>(mix: V<N>, p0: V<N>, p1: V<N>) -> V<N> {
    p1 * mix.clone() + p0 * (-mix + 1.)
}

fn matmul<const N0: usize, const N1: usize>(lns: M<N0, N1>, rhs: V<N1>) -> V<N0> {
    let rhs = rhs.reshape::<Rank2<N1, 1>>();
    let res: M<N0, 1> = lns.matmul(rhs);
    res.reshape()
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

/// tested with N_EMBED=1024 N_LAYER=24
// 0.000000049 to 0.000000063
pub static mut NUMPY_MAGIC_EPSILON: f32 = 0.000000053;

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
        
        (x.clone() - x.clone().mean().broadcast()) * w / sqrt(x.var() + unsafe { NUMPY_MAGIC_EPSILON }).broadcast() + b
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

fn summarize<const N: usize>(x: &V<N>) -> String {
    format!(
        "[{}, {}, {}, ..., {}, {}, {}]",
        x[[0]],
        x[[1]],
        x[[2]],
        x[[N - 3]],
        x[[N - 2]],
        x[[N - 1]]
    )
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

    pub fn load_safetensors<P: AsRef<Path>>(&mut self, path: P) -> Result<(), Error> {
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
        trace!(?token, "token");
        let x = self.emb.clone().select(dev.tensor(token));
        trace!(x = summarize(&x), "after emb");
        let x = self.ln_in.layer_norm(x);
        trace!(x = summarize(&x), "after ln_in");

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
                trace!(x = summarize(&x_), "after ln1");
                let (dx, att_state) = block.att.forward(x_, att_state);
                let x = x + dx;
                trace!(x = summarize(&x), "after att");

                let x_ = block.ln2.layer_norm(x.clone());
                trace!(x = summarize(&x_), "after ln2");
                let (dx, ffn_state) = block.ffn.forward(dev, x_, ffn_state);
                let x = x + dx;
                trace!(x = summarize(&x), "after ffn");

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
        trace!(x = summarize(&x), "after ln_out");
        let x = matmul(self.head.clone(), x); // "attention" head
        trace!(x = summarize(&x), "after head");

        // softmax but not quite
        let probs = x.softmax();

        // let mut sorted_probs = probs.array();
        // sorted_probs.sort_by(|a, b| b.total_cmp(a));
        // trace!(probs = summarize(&probs), "sorted_probs={:?}", &sorted_probs[0..10]);

        let a = probs.array();
        let sorted_ids = argsort_desc(&a);
        let max10_ids: Vec<usize> = sorted_ids
            .into_iter()
            .enumerate()
            .filter(|(_, shi)| *shi < 10)
            .map(|(i, _)| i)
            .collect();
        trace!(?max10_ids);

        (probs, state)
    }
}

/// np.argsort + reverse
pub fn argsort_desc<T: PartialOrd>(data: &[T]) -> Vec<usize> {
    let mut indices = (0..data.len()).collect::<Vec<_>>();
    indices.sort_by(|&j, &i| data[i].partial_cmp(&data[j]).unwrap());
    indices
}

/// Get the token
///
/// # Params
/// top_p: only select `top_p` percentile (0 to 1)
///
/// # Recommended values
/// temperature=1.0
/// top_p=0.85
pub fn sample_token<const N_VOCAB: usize>(
    dev: &Cpu,
    rng: &mut impl rand::Rng,
    probs: V<N_VOCAB>,
    temperature: f32,
    top_p: f32,
) -> usize {
    let mut sorted_probs: [f32; N_VOCAB] = probs.array();
    sorted_probs.sort_by(|a, b| b.partial_cmp(a).unwrap());

    let mut cum: [f32; N_VOCAB] = [0.; N_VOCAB];
    for i in 0..cum.len() {
        if i == 0 {
            cum[i] = sorted_probs[i];
        } else {
            cum[i] = sorted_probs[i] + cum[i - 1];
        }
    }

    let cutoff_i = cum
        .iter()
        .enumerate()
        .filter(|(_, x)| *x > &top_p)
        .max_by_key(|(i, _)| *i)
        .map(|(i, _)| i)
        .unwrap();
    let cutoff = sorted_probs[cutoff_i];

    let probs = probs
        .lt(&dev.tensor(cutoff).broadcast())
        .choose(dev.tensor(0.).broadcast(), probs);

    let probs = powf(probs, 1.0 / temperature);

    let sum_p: f32 = probs.clone().sum().array();

    let cutoff = rng.sample(Uniform::new(0., 1.)) * sum_p;

    let mut acc = 0.;
    let array = probs.array();
    for (i, p) in array.iter().enumerate() {
        acc += p;
        if acc > cutoff {
            return i;
        }
    }
    unreachable!("acc={acc} {array:?}")
}

#[derive(Debug, thiserror::Error)]
#[error("{0}")]
pub enum Error {
    SafeTensorError(#[from] ::safetensors::SafeTensorError),
    #[error("tensor size mismatch, expected={expected:?} actual={actual:?}")]
    MismatchedDimension {
        expected: Vec<usize>,
        actual: Vec<usize>,
    },
    IoError(#[from] std::io::Error),
}

fn _cp_block<const N_EMBED: usize, const N_EMBED_TIMES_4: usize>(
    xs: &SafeTensors,
    prefix: &str,
    block: &mut RWKVBlock<N_EMBED, N_EMBED_TIMES_4>,
) -> Result<(), Error> {
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

fn _cp_ln<const N: usize>(xs: &SafeTensors, prefix: &str, ln_in: &mut LN<N>) -> Result<(), Error> {
    _cp(xs, &format!("{prefix}.weight"), &mut ln_in.weight)?;
    _cp(xs, &format!("{prefix}.bias"), &mut ln_in.bias)?;
    Ok(())
}

/// copy tensor
fn _cp<S: Shape, D: CopySlice<f32>, T>(
    tensors: &SafeTensors,
    tensor_name: &str,
    tensor: &mut Tensor<S, f32, D, T>,
) -> Result<(), Error> {
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
