use std::path::Path;

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};

pub struct BMP {
    file: File,
    filesize: u32,
    offset: u32,
    width: u16,
    height: u16,
    image_data: Vec<u8>,
}

impl BMP {
    pub fn open(path: &Path) -> std::io::Result<BMP> {
        let mut file = OpenOptions::new().read(true).write(true).open(path)?;

        file.seek(SeekFrom::Start(0x02))?;
        let filesize = file.read_u32::<LittleEndian>()?;
        file.seek(SeekFrom::Start(0x0a))?;
        let offset = file.read_u32::<LittleEndian>()?;
        assert_eq!(
            file.read_u32::<LittleEndian>()?,
            12,
            "Поддерживается только BMP v2"
        );
        let width = file.read_u16::<LittleEndian>()?;
        let height = file.read_u16::<LittleEndian>()?;
        file.seek(SeekFrom::Current(2))?;
        assert_eq!(
            file.read_u16::<LittleEndian>()?,
            24,
            "Поддерживается только True Color"
        );
        let mut image_data = vec![0u8; width as usize * height as usize * 3];
        file.seek(SeekFrom::Start(offset as u64))?;
        file.read_exact(&mut image_data)?;

        Ok(BMP {
            file,
            filesize,
            offset,
            width,
            height,
            image_data,
        })
    }

    fn set_filesize(&mut self, new_filesize: u32) -> std::io::Result<()> {
        self.file.seek(SeekFrom::Start(0x02))?;
        self.file.write_u32::<LittleEndian>(new_filesize)?;
        self.filesize = new_filesize;
        self.file.set_len(new_filesize as u64)?;
        Ok(())
    }

    fn set_new_dimensions(&mut self, new_width: u16, new_heigth: u16) -> std::io::Result<()> {
        self.file.seek(SeekFrom::Start(0x12))?;
        self.file.write_u16::<LittleEndian>(new_width)?;
        self.file.write_u16::<LittleEndian>(new_heigth)?;
        self.width = new_width;
        self.height = new_heigth;
        Ok(())
    }

    fn calc_line_length(&self) -> u16 {
        ((self.width * 3) as f64 / 4.).ceil() as u16 * 4
    }

    pub fn resize(&mut self, coeff: f64) -> std::io::Result<()> {
        let new_width = (self.width as f64 * coeff).round() as u16;
        let new_height = (self.height as f64 * coeff).round() as u16;
        let mut new_image_data = Vec::with_capacity(new_width as usize * 3 * new_height as usize);
        for line in 0..new_height {
            for col in 0..new_width {
                let start_idx = (line as f64 / coeff) as usize * self.calc_line_length() as usize
                    + (col as f64 / coeff * 3.) as usize;
                new_image_data.extend_from_slice(&self.image_data[start_idx..start_idx + 3]);
            }
        }
        self.write_new_image_data(&new_image_data)?;
        self.set_filesize(
            self.filesize - self.image_data.len() as u32 + new_image_data.len() as u32,
        )?;
        self.set_new_dimensions(new_width, new_height)?;
        Ok(())
    }

    fn write_new_image_data(&mut self, image_data: &Vec<u8>) -> std::io::Result<()> {
        self.file.seek(SeekFrom::Start(self.offset as u64)).unwrap();
        self.file.write_all(&image_data)?;
        Ok(())
    }

    pub fn add(&mut self, other: &Self) -> std::io::Result<()> {
        assert!(
            other.height <= self.height && other.width <= self.width,
            "Невозможно вписать большее изображение в меньшее"
        );
        let bg_b = other.image_data[0];
        let bg_g = other.image_data[1];
        let bg_r = other.image_data[2];
        for line in 0..other.height as usize {
            for col in 0..other.width as usize {
                let other_idx = line * other.calc_line_length() as usize + col * 3;
                let self_idx = line * self.calc_line_length() as usize + col * 3;
                if other.image_data[other_idx] != bg_b
                    || other.image_data[other_idx + 1] != bg_g
                    || other.image_data[other_idx + 2] != bg_r
                {
                    self.image_data[self_idx] =
                        other.image_data[other_idx as usize] / 2 + self.image_data[self_idx] / 2;
                    self.image_data[self_idx + 1] =
                        other.image_data[other_idx + 1] / 2 + self.image_data[self_idx + 1] / 2;
                    self.image_data[self_idx + 2] =
                        other.image_data[other_idx + 2] / 2 + self.image_data[self_idx + 2] / 2;
                }
            }
        }
        self.write_new_image_data(&self.image_data.clone())?;
        Ok(())
    }

    pub fn add_text(&mut self, txt: &String) {
        let mut image_data_it = self.image_data.iter_mut();
        let mut len = txt.len();
        for _ in 0..32 {
            let byte = image_data_it.next().unwrap();
            *byte &= 0b11111100 | len as u8;
            len >>= 2;
        }

        for mut byte in txt.bytes() {
            for _ in 0..4 {
                let img_byte = image_data_it.next().unwrap();
                *img_byte &= 0b11111100 | byte;
                byte >>= 2;
            }
        }
        self.write_new_image_data(&self.image_data.clone()).unwrap();
    }

    pub fn read_text(&self) -> String {

    }
}
