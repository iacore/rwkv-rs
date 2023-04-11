use dfdx::prelude::*;

fn main() -> anyhow::Result<()> {
    let dev: Cpu = Default::default(); // or `Cpu`
                                       // let mlp = dev.build_module::<RWKV, f32>();
                                       // let a = &mlp.0.0;
                                       // let b = &a.weight;

    // let x: Tensor<Rank1<10>, f32, Cpu> = dev.zeros();
    // let y: Tensor<Rank1<2>, f32, Cpu> = mlp.forward(x);
    Ok(())
}
