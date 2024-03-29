# Задания для лабоpатоpных pабот по куpсу ПГИ

Требование к отчетам – титульный лист, задание, текст программы, скриншоты с результатами работы программы.

1. Пpеобpазование цветного BMP файла в чеpно-белый (найти в файле палитру, пpеобpазовать ее, усреднив по тройкам RGB
   цветов и записать получившийся файл под новым именем). Вывести основные характеристики BMP изображения (Работа с
   заголовком и палитрой).
2. Пpеобpазовать BMP файл, создав вокpуг него pамку из пикселей случайного цвета. Шиpина рамки - 15 пикселей (Работа с
   pастpовыми данными)
3. Пpеобpазовать BMP файл, pазвеpнув pастp на 90 градусов.
4. Вывести на экран 16, 256-цветный и True Color BMP файл.
5. Преобразовать 256-цветный ВМР файл используя коэффициент масштабирования от 0.1 до 10.
6. Написать программу для вписывания логотипа в BMP файлы. (Логотип создать в отдельном файле). Убрать фон логотипа и
   сделать логотип полупрозрачным.
7. Стеганография. Записать в младшие биты True Color BMP файла (контейнера) текстовый файл. Размер текстового файла
   взять 10% от размера графического. Извлечь файл из контейнера. Сравнить файлы.
8. Вывести на экpан PCX файл

# Запуск лаб

Для запуска лаб необходим Rust и Cargo. Их можно установить при помощи [Rustup](https://rustup.rs).

Также вместо того, чтобы искать подходящие файлы нужных форматов, рекомендую
установить [ImageMagick](https://imagemagick.org/script/download.php).

Возможно, в следующих примерах вместо вызова `convert` необходимо будет явно вызывать ImageMagick: `magick convert`.

## Lab 1

Для данной лабораторной необходим .bmp файл с таблицей цветов. Сконвертируем в него изображение (`.jpg` в примерах
выбран для наглядности. ImageMagick работает со многими другими форматами).

    $ convert img.jpg -type Palette img.bmp

Программа принимает в качестве аргементов оригинальный .bmp файл с палитрой и имя нового файла:

    $ cargo run --release --bin lab1 img.bmp decolorized.bmp

## Lab2

Работать с отдельными пикселями проще всего в формате CORE (BMP2). Также, чтобы рамка была видна, итоговое изображение
должно быть небольшого размера, поэтому уменьшим его:

    $ convert img.jpg -resize 800x600\> BMP2:img.bmp

После этого запускаем программу с аргументами:

    $ cargo run --release --bin lab2 img.bmp img_with_frame.bmp

## Lab3

Так же как и в прошлый раз, конвертируем в формат BMP2.

    $ convert img.jpg BMP2:img.bmp

Запускаем программу:

    $ cargo run --release --bin lab3 img.bmp rotated.bmp

## Lab4 (Linux only)

Для получения возможности вывода изобрания в tty необходимо добавить себя в группу `video`. После добавления потребуется
сделать logout-login.

    # gpasswd --add <username> video

Подготавливаем 3 изображения в формате BMP2 с разным количеством цветов и уменьшаем его, если оно больше размера экрана:

    $ convert img.jpg -resize 1920х1080\> BMP2:TrueColor.bmp
    $ convert img.jpg -resize 1920х1080\> -colors 256 BMP2:256.bmp
    $ convert img.jpg -resize 1920х1080\> -colors 16 BMP2:16.bmp

Переходим в любой TTY: `ctrl` + `alt` + `F3`

Запускаем программу для каждого файла. Выход происходит по нажатию `Enter`.

    $ cargo run --release --bin lab4 TrueColor.bmp
    $ cargo run --release --bin lab4 256.bmp
    $ cargo run --release --bin lab4 16.bmp

(Опционально) Выход из группы `video`:

    # gpasswd --delete <username> video

## Lab5

Конвертируем изображение в BMP2:

    $ convert img.jpg BMP2:img.bmp

Запускаем программу, передавая в качестве аргументов имя оригинального файла, имя нового файла и коэффициент
масштабирования:

    $ cargo run --release --bin lab5 img.bmp resized.bmp 0.7

## Lab6

Конвертируем изображение и логотип в BMP2:

    $ convert img.jpg BMP2:img.bmp
    $ convert logo.jpg BMP2:logo.bmp

Запускаем программу, со следующими аргументами: имя оригинального файла, имя файла с логотипом и имя нового файла:

    $ cargo run --release --bin lab6 img.bmp logo.bmp result.bmp

## Lab7

**Работает крайне нестабильно. Мб потом пофикшу, если не будет лень.**

Конвертируем изображение в BMP2:

    $ convert img.jpg BMP2:img.bmp

Запускаем программу, со следующими аргументами: имя оригинального файла, имя файла с текстом и имя нового файла:

    $ cargo run --release --bin lab7 img.bmp txt.txt result.bmp

## Lab8

**Ширина и высота изображения не считываются из файла.** Мне лень было разбироваться, где его там достать, поэтому
программа работает только с разрешением изображения 1440*1080.

**Необходимо вступить в группу Video (так же как в 4 лабе).**

Конвертируем изображение с выключенным RLE сжатием:

    $ convert img.jpg -compress None -resize 1920x1080\> img.pcx

Запускаем программу:

    $ cargo run --release --bin lab8 img.pcx
