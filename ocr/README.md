# OCR

We use MMOCR <https://github.com/open-mmlab/mmocr>

## Setup

After my failed local installation I presume you need cuda on your device.
The setup is already done in the python notebook. Else refer to <https://github.com/open-mmlab/mmocr>

## Models

There are a number of different Detection and Recognition models that can be used with MMOCR.

For recognition I found _ABINET_ performed the best after inspection of how it performed on a few samples .

For text detection it was not as clear. For now I used _DB_r18_ as this gives rectangular boxes, so simpler shapes than mother models
