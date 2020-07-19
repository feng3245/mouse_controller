from face_detection import face_detection
from head_pose_estimation import head_pose_estimation
from facial_landmarks_detection import facial_landmarks_detection
from gaze_estimation import gaze_estimation

fdmodel = face_detection('../models/face-detection-adas-binary-0001/FP32-INT1/face-detection-adas-binary-0001')
fdmodel.load_model()
fdmodel.check_model()
hpemodel = head_pose_estimation('../models/head-pose-estimation-adas-0001/FP32/head-pose-estimation-adas-0001')
hpemodel.load_model()
hpemodel.check_model()
flmodel = facial_landmarks_detection('../models/landmarks-regression-retail-0009/FP32/landmarks-regression-retail-0009')
flmodel.load_model()
flmodel.check_model()
gemodel = gaze_estimation('../models/gaze-estimation-adas-0002/FP32/gaze-estimation-adas-0002')
gemodel.load_model()
gemodel.check_model()
