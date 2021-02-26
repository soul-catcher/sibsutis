use framebuffer::{Framebuffer, KdMode};
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};


fn read_pixel_data(file: &mut File, width: u16, height: u16) -> std::io::Result<Vec<Vec<u8>>> {
    let mut pixel_data = vec![];
    for _ in 0..height * 3 {
        let mut buf = vec![0; width as usize];
        file.read_exact(&mut buf)?;
        pixel_data.push(buf);
    }
    let mut new_buf = vec![vec![0; width as usize * 3]; height as usize];
    for i in 0..height as usize {
        for j in 0..width as usize {
            new_buf[i][j * 3] = pixel_data[i * 3 + 2][j];
            new_buf[i][j * 3 + 1] = pixel_data[i * 3 + 1][j];
            new_buf[i][j * 3 + 2] = pixel_data[i * 3][j];
        }
    }
    Ok(new_buf)
}

fn draw_to_tty(pixel_data: &Vec<Vec<u8>>) {
    let mut framebuffer = Framebuffer::new("/dev/fb0").unwrap();

    let screen_height = framebuffer.var_screen_info.yres;
    let line_length = framebuffer.fix_screen_info.line_length;
    let mut frame = vec![0u8; (line_length * screen_height) as usize];
    for (i, line) in pixel_data.iter().enumerate() {
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

    let mut file = File::open(&args[1]).unwrap();
    file.seek(SeekFrom::Start(128)).unwrap();
    let pixel_data = read_pixel_data(&mut file, 1440, 1080).unwrap();
    draw_to_tty(&pixel_data);
}
