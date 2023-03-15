from mmocr.apis import MMOCRInferencer


def ocr(img, visualize=True):
    """
    Detect and recognize the characters in the image
    :param img: numpy img to do ocr on
    :param visualize: bool if to return image with visualization in results dict
    :return: ocr_results_dict with two keys: 'predictions' what we care about
     and 'visualization' the image for debugging/understanding
    """
    ocr_model = MMOCRInferencer(det='DB_r18', recog='ABINet')

    ocr_results = {}

    # MMOCR seems to throw error if no text detected
    try:
        ocr_results = ocr_model(img, return_vis=visualize)
    except IndexError:
        print("nothing detected")

    return ocr_results
