use std::env;

use anyhow::Context;
use dfdx::prelude::*;
use rwkv_rs::{sample_token, RWKVState, RWKV_430m};
use tokenizers::Tokenizer;

fn main() -> anyhow::Result<()> {
    if let Ok(n) = env::var("NUMPY_MAGIC") {
        let n: f32 = n.parse()?;
        unsafe { rwkv_rs::NUMPY_MAGIC_EPSILON = n };
    }

    // todo: arg
    // magic, temp, top_p

    let tokenizer = Tokenizer::from_file("20B_tokenizer.json").unwrap();

    //let prompt = "In a shocking finding, scientist discovered a herd of dragons living in a remote, previously unexplored valley, in Tibet. Even more surprising to the researchers was the fact that the dragons spoke perfect Chinese.";
    let prompt = "In a shocking finding, scientist discovered";

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

    let mut rng = rand::rngs::OsRng::default();
    
    use std::io::Write;
    let mut stdout = std::io::stdout();
    write!(stdout, "{prompt}")?;
    stdout.flush()?;


    for i in 0..16 {
        let Some(probs_taken) = probs.take() else { panic!() };
        let token_id = sample_token(&dev, &mut rng, probs_taken, 1.0, 0.8);
        // end of text
        if token_id == 0 {
            break;
        }
        let word = tokenizer.decode(vec![token_id as u32], true).unwrap();
        write!(stdout, "{word}")?;
        stdout.flush()?;

        let (probs_, state_) = model.forward(&dev, token_id, state.clone());
        state = state_;
        probs = Some(probs_);
    }
    println!();

    Ok(())
}
