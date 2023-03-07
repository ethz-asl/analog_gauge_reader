from mmocr.ocr import MMOCR


def ocr(img, img_out_dir=None):
    ocr_model = MMOCR(det='DB_r18', recog='ABINet')

    ocr_results = None

    # MMOCR seems to throw error if no text detected
    try:
        ocr_results = ocr_model.readtext(img, img_out_dir=img_out_dir)
    except IndexError:
        print("nothing detected")

    return ocr_results
