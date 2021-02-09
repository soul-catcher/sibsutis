use byteorder::{LittleEndian, ReadBytesExt};
use framebuffer::{Framebuffer, KdMode};
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};

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

fn draw_to_tty(pixel_data: Vec<Vec<u8>>) {
    let mut framebuffer = Framebuffer::new("/dev/fb0").unwrap();

    let screen_height = framebuffer.var_screen_info.yres;
    let line_length = framebuffer.fix_screen_info.line_length;
    let mut frame = vec![0u8; (line_length * screen_height) as usize];
    for (i, line) in pixel_data.iter().rev().enumerate() {
        for (j, pix) in line.iter().enumerate() {
            frame[i * line_length as usize + j + j / 3] = *pix;
        }
    }
    Framebuffer::set_kd_mode(KdMode::Graphics).unwrap();
    framebuffer.write_frame(&frame);
    std::io::stdin().read_line(&mut String::new()).unwrap();
    Framebuffer::set_kd_mode(KdMode::Text).unwrap();
}

fn main() {
    let args: Vec<_> = std::env::args().collect();
    let in_file = &args[1];

    let mut file = File::open(in_file).unwrap(); // TODO изменить на просто READ

    file.seek(SeekFrom::Start(0x0a)).unwrap();
    let off_bits = file.read_u32::<LittleEndian>().unwrap();

    file.seek(SeekFrom::Start(0x0e)).unwrap();
    let info_table_size = file.read_u32::<LittleEndian>().unwrap();
    assert_eq!(
        info_table_size, 12,
        "Поддерживается только файл формата BMP2"
    );

    file.seek(SeekFrom::Start(0x18)).unwrap();
    let bit_count = file.read_u16::<LittleEndian>().unwrap();
    dbg!(bit_count);

    file.seek(SeekFrom::Start(0x12)).unwrap();
    let width = file.read_u16::<LittleEndian>().unwrap();
    let height = file.read_u16::<LittleEndian>().unwrap();
    dbg!(width, height);

    file.seek(SeekFrom::Start(off_bits as u64)).unwrap();
    let pixel_data = read_pixel_data(&mut file, width, height).unwrap();
    draw_to_tty(pixel_data);
}
