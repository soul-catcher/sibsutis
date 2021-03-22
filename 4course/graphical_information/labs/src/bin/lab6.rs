use graphical_information::BMP;
use std::fs;
use std::path::Path;

fn main() {
    let args: Vec<_> = std::env::args().collect();
    let in_file = &args[1];
    let logo = &args[2];
    let out_file = &args[3];
    fs::copy(in_file, out_file).unwrap();
    let mut bmp = BMP::open(Path::new(out_file)).unwrap();
    let logo = BMP::open(Path::new(logo)).unwrap();
    bmp.add(&logo).unwrap();
}