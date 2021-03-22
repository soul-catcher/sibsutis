use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use std::fs;
use std::io::{Seek, SeekFrom};

#[derive(Debug)]
struct Header {
    bf_type: u16,
    bf_size: u32,
    bf_reserved1: u16,
    bf_reserved2: u16,
    bf_off_bits: u32,
}

impl Header {
    fn from_reader(rdr: &mut impl std::io::Read) -> std::io::Result<Self> {
        let bf_type = rdr.read_u16::<LittleEndian>()?;
        let bf_size = rdr.read_u32::<LittleEndian>()?;
        let bf_reserved1 = rdr.read_u16::<LittleEndian>()?;
        let bf_reserved2 = rdr.read_u16::<LittleEndian>()?;
        let bf_off_bits = rdr.read_u32::<LittleEndian>()?;
        Ok(Header {
            bf_type,
            bf_size,
            bf_reserved1,
            bf_reserved2,
            bf_off_bits,
        })
    }
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
    let header = Header::from_reader(&mut file)?;
    dbg!(&header);

    let info_table_size = file.read_u32::<LittleEndian>()?;
    assert_ne!(info_table_size, 12, "Версия таблицы CORE не поддерживается");

    dbg!(info_table_size);
    file.seek(SeekFrom::Start(0x2e))?;
    let color_table_size = file.read_u32::<LittleEndian>()?;
    assert_ne!(color_table_size, 0, "В файле отсутствует таблица цветов");
    dbg!(color_table_size);
    file.seek(SeekFrom::Start(0x8a))?;

    for _ in 0..color_table_size {
        let blue = file.read_u8()?;
        let green = file.read_u8()?;
        let red = file.read_u8()?;
        let average = blue / 3 + green / 3 + red / 3;
        file.seek(SeekFrom::Current(-3))?;
        file.write_u8(average)?;
        file.write_u8(average)?;
        file.write_u8(average)?;
        file.seek(SeekFrom::Current(1))?;
    }
    Ok(())
}
