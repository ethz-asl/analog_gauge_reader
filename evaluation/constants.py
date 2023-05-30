# Filenames---------------------------------------------------------
RESULT_FILE_NAME = "result.json"
RESULT_FULL_FILE_NAME = "result_full.json"
ERROR_FILE_NAME = "error.json"

ORIGINAL_IMG_FILE_NAME = "original_image.jpg"

# Keys in Json Files-------------------------------------------------
READING_KEY = "reading"
RANGE_KEY = "range"
MEASURE_UNIT_KEY = "unit"

OCR_NONE_DETECTED_KEY = "No OCR reading with a number"
OCR_ONLY_ONE_DETECTED_KEY = "Only found 1 number with ocr"
NOT_AN_ELLIPSE_ERROR_KEY = "Error, ellipse has faulty parameters"
SEGMENTATION_FAILED_KEY = "Segmentation failed"
NEEDLE_ELLIPSE_NO_INTERSECT = "Needle line and ellipse do not intersect"

IMG_SIZE_KEY = "image size"
OCR_NUM_KEY = "OCR Numbers"
OCR_UNIT_KEY = "OCR Unit"
GAUGE_DET_KEY = "Gauge"
KEYPOINT_START_KEY = "Start Notch"
KEYPOINT_END_KEY = "End Notch"
KEYPOINT_NOTCH_KEY = "Keypoint Notch"
ORIGINAL_IMG_KEY = "original image"

PRED = 'prediction'
TRUTH = 'true_reading'
ABS_ERROR = 'total absolute error'
REL_ERROR = 'total relative error'
N_FAILED = 'number of failed predictions'
N_FAILED_OCR = 'number of failed OCR, less than 2 numbers detected'
N_FAILED_NO_ELLIPSE = 'number of examples, where ellipse has faulty parameters'
N_FAILED_SEG = 'number of examples, where needle segmenatation failed'
N_FAILED_ELLIPSE_LINE_NO_INTERSECT = 'number of examples, where ellipse has faulty parameters'

# Other constants-------------------------------------------------------
FAILED = 'Failed'

UNIT_LIST = ['bar', 'mbar', 'psi', 'MPa']
