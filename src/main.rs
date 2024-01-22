pub fn main() {
    let ds = rusty_llms::HuggingFaceDataset::load_dataset("nuprl/CanItEdit", "test").unwrap();
    for ex in ds.into_iter() {
        let before: String = ex.get("before").unwrap();
        println!("{}", before);
    }
}
