use anyhow::Context;
use dfdx::prelude::*;
use rwkv_rs::{sample_token, RWKVState, RWKV_430m};
use tokenizers::Tokenizer;

fn main() -> anyhow::Result<()> {
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

    let mut rng = rand::thread_rng();

    print!("{prompt}");
    loop {
        if let Some(probs_taken) = probs.take() {
            let token_id = sample_token(&mut rng, probs_taken, 1.0, 0.85);
            // end of text
            if token_id == 0 {
                break;
            }
            let word = tokenizer.decode(vec![token_id as u32], false).unwrap();
            print!("{word}");

            let (probs_, state_) = model.forward(&dev, token_id, state.clone());
            state = state_;
            probs = Some(probs_);
        }
    }

    Ok(())
}
