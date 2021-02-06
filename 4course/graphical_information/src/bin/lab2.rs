use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use rand::{thread_rng, Rng};
use std::fs;
use std::io::{Read, Seek, SeekFrom, Write};

fn read_pixel_data(
    file: &mut impl Read,
    width: u16,
    height: u16,
) -> std::io::Result<Vec<Vec<[u8; 3]>>> {
    let mut pixel_data = vec![];
    for _ in 0..height {
        let mut line_vec = vec![];
        for _ in 0..width {
            line_vec.push([file.read_u8()?, file.read_u8()?, file.read_u8()?])
        }
        pixel_data.push(line_vec);
    }
    Ok(pixel_data)
}

fn write_random_pixels(file: &mut impl Write, n: usize) -> std::io::Result<()> {
    let mut rng = thread_rng();
    for _ in 0..n {
        file.write_u8(rng.gen())?;
        file.write_u8(rng.gen())?;
        file.write_u8(rng.gen())?;
    }
    Ok(())
}

fn write_random_lines(file: &mut impl Write, line_len: usize, n: usize) -> std::io::Result<()> {
    for _ in 0..n {
        write_random_pixels(file, line_len)?;
        for _ in 0..(4 - line_len * 3 % 4) % 4 {
            file.write_u8(0)?;
        }
    }
    Ok(())
}

fn write_pixel_data(file: &mut impl Write, pixel_data: Vec<Vec<[u8; 3]>>) -> std::io::Result<()> {
    let additional_bytes = (4 - (pixel_data[0].len() + 30) * 3 % 4) % 4;

    write_random_lines(file, pixel_data[0].len() + 30, 15)?;

    for line in &pixel_data {
        write_random_pixels(file, 15)?;

        for pixel in line {
            file.write_u8(pixel[0])?;
            file.write_u8(pixel[1])?;
            file.write_u8(pixel[2])?;
        }
        write_random_pixels(file, 15)?;
        for _ in 0..additional_bytes {
            file.write_u8(0)?;
        }
    }
    write_random_lines(file, pixel_data[0].len() + 30, 15)?;

    Ok(())
}

fn main() -> std::io::Result<()> {
    let args: Vec<_> = std::env::args().collect();
    let in_file = &args[1];
    let out_file = &args[2];
    fs::copy(in_file, out_file)?;

    let mut file = fs::OpenOptions::new()
        .read(true)
        .write(true)
        .open(out_file)?;

    file.seek(SeekFrom::Start(0x0a))?;
    let off_bits = file.read_u32::<LittleEndian>()?;

    file.seek(SeekFrom::Start(0x0e))?;
    let info_table_size = file.read_u32::<LittleEndian>()?;
    assert_eq!(
        info_table_size, 12,
        "Поддерживается только файл формата BMP2"
    );

    file.seek(SeekFrom::Start(0x18))?;
    let bit_count = file.read_u16::<LittleEndian>()?;
    assert_eq!(bit_count, 24, "Файл имеет цвет, отличный от 16-битного");

    file.seek(SeekFrom::Start(0x12))?;
    let width = file.read_u16::<LittleEndian>()?;
    let height = file.read_u16::<LittleEndian>()?;
    dbg!(width, height);

    file.seek(SeekFrom::Start(off_bits as u64))?;
    let pixel_data = read_pixel_data(&mut file, width, height)?;
    file.seek(SeekFrom::Start(off_bits as u64))?;
    write_pixel_data(&mut file, pixel_data)?;

    file.seek(SeekFrom::Start(0x12))?;
    file.write_u16::<LittleEndian>(width + 30)?;
    file.write_u16::<LittleEndian>(height + 30)?;
    file.seek(SeekFrom::Start(0x18))?;

    Ok(())
}
// TODO оптимизировать
