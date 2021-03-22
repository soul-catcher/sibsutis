use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use std::fs;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom, Write};

fn read_pixel_data(file: &mut File, width: u16, height: u16) -> std::io::Result<Vec<Vec<u8>>> {
    let mut pixel_data = vec![];
    for _ in 0..height {
        let mut buf = vec![0; width as usize * 3];
        file.read_exact(&mut buf)?;
        pixel_data.push(buf);
        file.seek(SeekFrom::Current(
            calc_additional_bytes(width as usize * 3) as i64
        ))?;
    }
    Ok(pixel_data)
}

fn calc_additional_bytes(bytes: usize) -> usize {
    (4 - bytes % 4) % 4
}

fn write_pixel_data(file: &mut impl Write, pixel_data: Vec<Vec<u8>>) -> std::io::Result<()> {
    let width = pixel_data[0].len() / 3;
    let height = pixel_data.len();
    let additional_bytes = calc_additional_bytes(height * 3);
    let mut new_pixel_data = vec![vec![0; height * 3]; width];
    for i in 0..height {
        for j in 0..width {
            new_pixel_data[j][i * 3] = pixel_data[i][j * 3];
            new_pixel_data[j][i * 3 + 1] = pixel_data[i][j * 3 + 1];
            new_pixel_data[j][i * 3 + 2] = pixel_data[i][j * 3 + 2];
        }
    }
    for line in new_pixel_data {
        file.write(&line)?;
        file.write(&vec![0; additional_bytes])?;
    }

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
    file.seek(SeekFrom::Start(0x12))?;
    file.write_u16::<LittleEndian>(height)?;
    file.write_u16::<LittleEndian>(width)?;

    file.seek(SeekFrom::Start(off_bits as u64))?;
    let pixel_data = read_pixel_data(&mut file, width, height)?;
    file.seek(SeekFrom::Start(off_bits as u64))?;
    write_pixel_data(&mut file, pixel_data)?;
    Ok(())
}
