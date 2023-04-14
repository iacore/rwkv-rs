use anyhow::Context;
use dfdx::prelude::*;
use rwkv_rs::{sample_token, RWKVState, RWKV_430m};
use std::io::Write;
use tokenizers::Tokenizer;
use tracing::instrument;

fn main() -> anyhow::Result<()> {
    // use tracing_subscriber::layer::SubscriberExt;

    // tracing::subscriber::set_global_default(
    //     tracing_subscriber::registry().with(tracing_tracy::TracyLayer::new()),
    // )
    // .expect("set up the subscriber");

    // if let Ok(n) = env::var("NUMPY_MAGIC") {
    //     let n: f32 = n.parse()?;
    //     unsafe { rwkv_rs::NUMPY_MAGIC_EPSILON = n };
    // }

    // todo: arg
    // magic, temp, top_p

    let tokenizer = Tokenizer::from_file("20B_tokenizer.json").unwrap();

    //let prompt = "In a shocking finding, scientist discovered a herd of dragons living in a remote, previously unexplored valley, in Tibet. Even more surprising to the researchers was the fact that the dragons spoke perfect Chinese.";
    let prompt = "In a shocking finding, scientist discovered";

    let dev: Cpu = Default::default();

    let mut state = RWKVState::default();
    let mut model = RWKV_430m::zeros(&dev);
    model
        .load_safetensors("RWKV-4-Pile-430M-20220808-8066.safetensors")
        .context("Load model")?;

    let encoded = tokenizer.encode(prompt, true).unwrap();
    let mut probs = None;
    for &token in encoded.get_ids() {
        let (probs_, state_) = model.forward(token as usize, state.clone());
        state = state_;
        probs = Some(probs_);
    }

    let mut rng = rand::rngs::OsRng::default();

    let mut stdout = std::io::stdout();
    write!(stdout, "{prompt}")?;
    stdout.flush()?;

    for _i in 0..1000 {
        let token_id = next_token(&mut probs, &mut rng);
        // end of text
        if token_id == 0 {
            break;
        }
        print_token(&tokenizer, token_id, &mut stdout)?;

        let (probs_, state_) = infer(&model, token_id, &state);
        state = state_;
        probs = Some(probs_);
    }
    println!();

    Ok(())
}

#[instrument(skip_all)]
fn infer(
    model: &rwkv_rs::RWKV<24, 1024, 4096, 50277>,
    token_id: usize,
    state: &[rwkv_rs::RWKVBlockState<1024>; 24],
) -> (
    Tensor<(Const<50277>,), f32, Cpu>,
    [rwkv_rs::RWKVBlockState<1024>; 24],
) {
    model.forward(token_id, state.clone())
}

#[instrument(skip_all)]
fn next_token(
    probs: &mut Option<Tensor<(Const<50277>,), f32, Cpu>>,
    rng: &mut rand::rngs::OsRng,
) -> usize {
    let Some(probs_taken) = probs.take() else { panic!() };
    let token_id = sample_token(rng, probs_taken, 1.0, 0.8);
    token_id
}

#[instrument(skip_all)]
fn print_token(
    tokenizer: &Tokenizer,
    token_id: usize,
    stdout: &mut std::io::Stdout,
) -> Result<(), anyhow::Error> {
    let word = tokenizer.decode(vec![token_id as u32], true).unwrap();
    write!(stdout, "{word}")?;
    stdout.flush()?;
    Ok(())
}
