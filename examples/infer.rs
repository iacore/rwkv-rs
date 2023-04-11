use dfdx::prelude::*;
use rwkv_rs::RWKV_430m;

use anyhow::Context;

fn main() -> anyhow::Result<()> {
    let dev: Cpu = Default::default();
    let mut model = RWKV_430m::zeros(&dev);
    model.load_safetensors("RWKV-4-Pile-430M-20220808-8066.safetensors").context("Load model")?;
    
    Ok(())
}
