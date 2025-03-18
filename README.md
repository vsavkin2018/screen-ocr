# screen-ocr

Эта программа предназначена для распознавания текста на скриншотах, присылаемых пользователями, особенно в Telegram например.
В данной версии поддерживаются только LLM через Ollama, планирую добавить и другие способы: tesseract, Vision.UI (?)


## Установка

1. Надо установить Ollama: https://github.com/ollama/ollama/blob/main/docs/linux.md

Там просто скачать архив и распаковать, можно запускать прям оттуда где распаковано, например `~/Build/ollama/bin/ollama serve` . 
Собственно эта команда должна запуститься и ждать запросов на http://localhost:11434. Оставить работать в отдельном терминале или как-то ещё.

2. Надо скачать какую-нибудь vision LLM.

Если есть GPU c 8G VRAM, то можно скачать `gemma3:4b`. Можно скачать `llama3.2-vision` - она побольше, не влезет в такую видеопамять, будет работать медленнее.
Модели можно искать на сайте ollama по слову vision например.

Сейчас в скрипте закодирована модель llama3.2-vision - конфиг пока прям внутри, планирую добавить отдельный конфиг и выбор моделей в рантайме. А пока так.

3. Рекомендую поставить терминал kitty, он нужен чтобы графический превью того что распознаётся был:
```
apt install kitty kitty-shell-integration kitty-terminfo
```

4. Создать питоновский venv и установить туда библиотеки:
```
python3 -m venv env
source env/bin/activate
pip install -U pip  && pip install httpx prompt_toolkit pillow
```

6. Можно запускать прям из терминала (kitty или другого): `python3 screenshot_ocr.py` - но я представляю так, что можно повесить на хоткей запуск 
kitty с этой программой, вот например запускать скрипт такой:
```
#!/bin/bash

cd <путь>/screen-ocr || exit 1

exec kitty -e ./env/bin/python3 screenshot_ocr.py
```

## Как работать

Программа берёт по умолчанию картинки типа `~/Pictures/Screenshots/Screenshot*.png` - это настраивается. Именно туда кладёт скриншоты Gnome3 под Debian или Ubuntu. То есть можно средствами Gnome выбрать область на экране, она сохранится в файл, и потом по хоткею запустить скрипт типа приведённого выше.

Программа открывает последний скриншот. Moжно кнопками n и p ходить циклически по скриншотам. Кнопкой o начинается OCR. Сначала может потормозить, а потом пойдёт текст. Ctrl+C может прервать процесс. Можно выделить текст стандартным образом мышкой и скопировать. Ctrl+Shift+C - kitty копирует в буфер обмена.

В конце нажать q для выхода.

Планируется больше фичей: выбор модели, выбор промпта, чат с моделью чтоб уточнить что ещё нужно. А пока так.
