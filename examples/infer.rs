use std::{fs::File, io::Write};

use anyhow::Context;
use dfdx::prelude::*;
use rwkv_rs::{sample_token, RWKVState, RWKV_430m};
use tokenizers::Tokenizer;
use tracing_subscriber::EnvFilter;

unsafe fn cast_slice_to_u8(data: &[f32]) -> &[u8] {
    std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4)
}

fn main() -> anyhow::Result<()> {
    let (non_blocking, _guard) = tracing_appender::non_blocking(File::create("target/rwkv.log")?);
    tracing_subscriber::fmt()
        .with_ansi(false)
        .without_time()
        .compact()
        .with_env_filter(EnvFilter::new("none,rwkv=trace"))
        .with_writer(non_blocking)
        .init();

    let tokenizer = Tokenizer::from_file("20B_tokenizer.json").unwrap();

    let prompt = "In a shocking finding, scientist discovered a herd of dragons living in a remote, previously unexplored valley, in Tibet. Even more surprising to the researchers was the fact that the dragons spoke perfect Chinese.";

    let dev: Cpu = Default::default();
    let mut state = RWKVState::zeros(&dev);
    let mut model = RWKV_430m::zeros(&dev);
    model
        .load_safetensors("RWKV-4-Pile-430M-20220808-8066.safetensors")
        .context("Load model")?;

    let encoded = tokenizer.encode(prompt, true).unwrap();

    let mut probs = None;
    for &token in encoded.get_ids() {
        let (probs_, state_) = model.forward(&dev, token as usize, state.clone());
        state = state_;
        probs = Some(probs_);
    }

    File::create("probs.raw")?
        .write_all(unsafe { cast_slice_to_u8(&probs.as_ref().unwrap().array()) })?;
    // File::create("state.ffn_state.raw")?
    //     .write_all(unsafe { cast_slice_to_u8(&state.ffn_state.array()) })?;

    let mut rng = rand::thread_rng();

    print!("{prompt}");
    loop {
        if let Some(probs_taken) = probs.take() {
            let token_id = sample_token(&dev, &mut rng, probs_taken, 1.0, 0.85);
            // end of text
            if token_id == 0 {
                break;
            }
            let word = tokenizer.decode(vec![token_id as u32], true).unwrap();
            print!("{word}");

            let (probs_, state_) = model.forward(&dev, token_id, state.clone());
            state = state_;
            probs = Some(probs_);
        }
    }

    Ok(())
}
