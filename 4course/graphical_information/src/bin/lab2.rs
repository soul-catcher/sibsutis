use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use rand::{thread_rng, RngCore};
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

fn write_random_pixels(file: &mut impl Write, n: usize) -> std::io::Result<()> {
    let mut rng = thread_rng();
    let mut bytes = vec![0; n * 3];
    rng.fill_bytes(&mut bytes);
    file.write(&bytes)?;
    Ok(())
}

fn write_random_lines(file: &mut impl Write, line_len: usize, n: usize) -> std::io::Result<()> {
    for _ in 0..n {
        write_random_pixels(file, line_len)?;
        for _ in 0..calc_additional_bytes(line_len * 3) {
            file.write_u8(0)?;
        }
    }
    Ok(())
}

fn calc_additional_bytes(bytes: usize) -> usize {
    (4 - bytes % 4) % 4
}

fn write_pixel_data(file: &mut impl Write, pixel_data: Vec<Vec<u8>>) -> std::io::Result<()> {
    let additional_bytes = calc_additional_bytes(pixel_data[0].len() + 30 * 3);

    write_random_lines(file, pixel_data[0].len() / 3 + 30, 15)?;

    for line in &pixel_data {
        write_random_pixels(file, 15)?;
        file.write(&line)?;
        write_random_pixels(file, 15)?;
        for _ in 0..additional_bytes {
            file.write_u8(0)?;
        }
    }
    write_random_lines(file, pixel_data[0].len() / 3 + 30, 15)?;
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
    assert_eq!(bit_count, 24, "Файл имеет цвет, отличный от 24-битного");

    file.seek(SeekFrom::Start(0x12))?;
    let width = file.read_u16::<LittleEndian>()?;
    let height = file.read_u16::<LittleEndian>()?;
    dbg!(width, height);

    file.seek(SeekFrom::Start(off_bits as u64))?;
    let pixel_data = read_pixel_data(&mut file, width, height)?;
    file.seek(SeekFrom::Start(off_bits as u64))?;
    write_pixel_data(&mut file, pixel_data)?;

    // Изменение параметров файла: задание нового размера в байтах и пикселях
    file.seek(SeekFrom::Start(0x12))?;
    file.write_u16::<LittleEndian>(width + 30)?;
    file.write_u16::<LittleEndian>(height + 30)?;
    file.seek(SeekFrom::Start(0x02))?;
    let filesize = file.read_u32::<LittleEndian>()?;
    let line_size = width as u32 * 3 + calc_additional_bytes(width as usize * 3) as u32;
    let new_line_size =
        (width as u32 + 30) * 3 + calc_additional_bytes((width + 30) as usize * 3) as u32;
    let new_filesize = filesize - line_size * height as u32 + new_line_size * (height + 30) as u32;
    dbg!(filesize, new_filesize, line_size, new_line_size);
    file.seek(SeekFrom::Start(0x02))?;
    file.write_u32::<LittleEndian>(new_filesize)?;
    Ok(())
}
