pub fn main() {
    let ds = rusty_llms::HuggingFaceDataset::load_dataset("nuprl/CanItEdit", "test");
    for ex in ds.into_iter() {
        println!("{:?}", ex);
    }

}
