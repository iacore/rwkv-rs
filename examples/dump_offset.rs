//! debugging only

use memmap2::MmapOptions;
use safetensors::SafeTensors;

fn main() -> Result<(), rwkv_rs::Error> {
    let f = std::fs::File::open("RWKV-4-Pile-430M-20220808-8066.safetensors")?;
    let buffer = unsafe { MmapOptions::new().map(&f)? };
    dbg!(buffer.as_ptr());
    // let _xs = SafeTensors::deserialize(&buffer)?;
    let (_n, metadata) = SafeTensors::read_metadata(&buffer)?;
    // SafeTensors::deserialize(buffer)
    for (_k, v) in metadata.tensors() {
        if v.data_offsets.0 % 4 != 0 || v.data_offsets.1 % 4 != 0 {
            dbg!(&v.data_offsets);
        }
    }

    let tensors = SafeTensors::deserialize(&buffer)?;
    for name in tensors.names() {
        let x = tensors.tensor(&name)?;
        let p = x.data().as_ptr();
        let p = p as usize;
        if p % 4 != 0 {
            dbg!(p);
        }
    }
    Ok(())
}
