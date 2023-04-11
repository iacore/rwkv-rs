use std::mem::zeroed;

use dfdx::prelude::*;

pub type V = Tensor1D<768>;
pub type M = Tensor2D<768, 768>;

fn mix(mix: V, p0: V, p1: V) -> V {
    p1 * mix.clone() + p0 * (-mix + 1.)
}

#[derive(Clone)]
pub struct ATT {
    pub decay: V, // time_decay
    pub bonus: V, // time_first
    pub mix_k: V, // time_mix_k
    pub mix_v: V, // time_mix_v
    pub mix_r: V, // time_mix_r
    pub wk: M,    // key.weight
    pub wv: M,    // value.weight
    pub wr: M,    // receptance.weight
    pub wout: M,  // output.weight
}

#[derive(Clone)]
pub struct ATTState {
    pub x: V,
    pub num: V,
    pub den: V,
}

impl ATT {
    pub fn forward(&self, x: V, last: ATTState) -> (V, ATTState) {
        let decay = self.decay.clone();
        let bonus = self.bonus.clone();
        let mix_k = self.mix_k.clone();
        let mix_v = self.mix_v.clone();
        let mix_r = self.mix_r.clone();
        let wk = self.wk.clone();
        let wv = self.wv.clone();
        let wr = self.wr.clone();
        let wout = self.wout.clone();

        let k = matmul(mix(mix_k, last.x.clone(), x.clone()), wk);
        let v = matmul(mix(mix_v, last.x.clone(), x.clone()), wv);
        let r = matmul(mix(mix_r, last.x, x.clone()), wr);

        let _0 = exp(bonus + k.clone());

        let wkv = (last.num.clone() + _0.clone() * v.clone()) / (last.den.clone() + _0);
        let rwkv = sigmoid(r) * wkv;

        let _1 = exp(-exp(decay));
        let _2 = exp(k);
        let num = _1.clone() * last.num + _2.clone() * v;
        let den = _1 * last.den + _2;

        (matmul(rwkv, wout), ATTState { x, num, den })
    }
}

#[derive(Clone)]
pub struct FFN {
    pub mix_k: V, // time_mix_k
    pub mix_r: V, // time_mix_r
    pub wk: M,    // key.weight
    pub wv: M,    // value.weight
    pub wr: M,    // receptance.weight
}

impl FFN {
    pub fn forward(&self, dev: &Cpu, x: V, last_x: V) -> (V, V) {
        let mix_k = self.mix_k.clone();
        let mix_r = self.mix_r.clone();
        let wk = self.wk.clone();
        let wv = self.wv.clone();
        let wr = self.wr.clone();

        let k = matmul(mix(mix_k, last_x.clone(), x.clone()), wk);
        let r = matmul(mix(mix_r, last_x, x.clone()), wr);
        let vk = matmul(maximum(k, dev.zeros()).square(), wv);

        (sigmoid(r) * vk, x)
    }
}

// fn time_mixing(
//     x: V,
//     last_x: V,
//     last_num: V,
//     last_den: V,
//     decay: V,
//     bonus: V,
//     mix_k: V,
//     mix_v: V,
//     mix_r: V,
//     wk: M,
//     wv: M,
//     wr: M,
//     wout: M,
// ) -> (V, (V, V, V)) {
//     let k = matmul(mix(mix_k, last_x.clone(), x.clone()), wk);
//     let v = matmul(mix(mix_v, last_x.clone(), x.clone()), wv);
//     let r = matmul(mix(mix_r, last_x, x.clone()), wr);

//     let _0 = exp(bonus + k.clone());

//     let wkv = (last_num.clone() + _0.clone() * v.clone()) / (last_den.clone() + _0);
//     let rwkv = sigmoid(r) * wkv;

//     let _1 = exp(-exp(decay));
//     let _2 = exp(k);
//     let num = _1.clone() * last_num + _2.clone() * v;
//     let den = _1 * last_den + _2;

//     (matmul(rwkv, wout), (x, num, den))
// }
