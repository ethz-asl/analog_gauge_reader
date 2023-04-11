from mmocr.apis import MMOCRInferencer
import matplotlib.pyplot as plt
import numpy as np

from ocr.ocr_reading import OCRReading


def ocr(img, visualize=True):
    """
    Detect and recognize the characters in the image
    :param img: numpy img to do ocr on
    :param visualize: bool if to return image with visualization in results dict
    :return: ocr_results_dict with two keys: 'predictions' what we care about
     and 'visualization' the image for debugging/understanding
    """
    ocr_model = MMOCRInferencer(det='DB_r18', rec='ABINet')

    readings = []

    # MMOCR seems to throw error if no text detected
    try:
        results = ocr_model(img, return_vis=visualize)

        if visualize:
            visualization = results['visualization'][0]
            plt.figure()
            plt.imshow(visualization)
            plt.show()

        polygons = results['predictions'][0]['det_polygons']

        shapes = []
        for coord_list in polygons:
            shape_array = np.array(coord_list)
            shape_array = shape_array.reshape(-1, 2)
            shapes.append(shape_array)

        scores = results['predictions'][0]['rec_scores']
        texts = results['predictions'][0]['rec_texts']

        assert len(scores) == len(texts) and len(scores) == len(shapes)

        for index, score in enumerate(scores):
            reading = OCRReading(shapes[index], texts[index], score)
            readings.append(reading)

    except IndexError:
        print("nothing detected")

    return readings
