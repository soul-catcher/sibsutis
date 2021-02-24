use graphical_information::BMP;
use std::fs;
use std::io::Read;
use std::path::Path;

fn main() {
    let args: Vec<_> = std::env::args().collect();
    let in_file = &args[1];
    let mut txt = String::new();
    fs::File::open(Path::new(&args[2]))
        .unwrap()
        .read_to_string(&mut txt).unwrap();
    let out_file = &args[3];
    fs::copy(in_file, out_file).unwrap();
    let mut bmp = BMP::open(Path::new(out_file)).unwrap();
    bmp.add_text(&txt);
    println!("{}", bmp.read_text());
}
