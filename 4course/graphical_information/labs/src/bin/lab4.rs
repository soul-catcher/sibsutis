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

fn read_pixel_data_color_pallete(
    file: &mut File,
    width: u16,
    heigth: u16,
    bit_count: u16,
) -> std::io::Result<Vec<Vec<u8>>> {
    let mut pixel_data = vec![];
    file.seek(SeekFrom::Start(0x1a))?;
    let mut color_table = vec![0; 2usize.pow(bit_count as u32) * 3];
    file.read_exact(&mut color_table)?;
    file.seek(SeekFrom::Start(0x0a))?;
    let off_bits = file.read_u32::<LittleEndian>()?;
    file.seek(SeekFrom::Start(off_bits as u64))?;
    for _ in 0..heigth {
        let mut buf = vec![0; ((width as usize * bit_count as usize) + 4) / 8];
        file.read_exact(&mut buf)?;
        file.seek(SeekFrom::Current(calc_additional_bytes(buf.len()) as i64))?;
        let mut new_buf = vec![];
        for i in 0..width {
            match bit_count {
                8 => {
                    new_buf.push(color_table[buf[i as usize] as usize * 3]);
                    new_buf.push(color_table[buf[i as usize] as usize * 3 + 1]);
                    new_buf.push(color_table[buf[i as usize] as usize * 3 + 2]);
                }
                4 => {
                    if i % 2 == 0 {
                        new_buf.push(color_table[(buf[i as usize / 2] >> 4) as usize * 3]);
                        new_buf.push(color_table[(buf[i as usize / 2] >> 4) as usize * 3 + 1]);
                        new_buf.push(color_table[(buf[i as usize / 2] >> 4) as usize * 3] + 2);
                    } else {
                        new_buf.push(color_table[(buf[i as usize / 2] & 0b1111) as usize * 3]);
                        new_buf.push(color_table[(buf[i as usize / 2] & 0b1111) as usize * 3 + 1]);
                        new_buf.push(color_table[(buf[i as usize / 2] & 0b1111) as usize * 3 + 2]);
                    }
                }
                _ => panic!("Битность {} не поддерживается", bit_count),
            };
        }
        pixel_data.push(new_buf);
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

    let mut file = File::open(in_file).unwrap();

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

    let pixel_data = match bit_count {
        24 => {
            file.seek(SeekFrom::Start(off_bits as u64)).unwrap();
            read_pixel_data(&mut file, width, height).unwrap()
        }
        8 | 4 => read_pixel_data_color_pallete(&mut file, width, height, bit_count).unwrap(),
        _ => panic!("Битность {} не поддерживается", bit_count),
    };
    draw_to_tty(pixel_data);
}
