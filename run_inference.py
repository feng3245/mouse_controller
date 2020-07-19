import sys
sys.path.append("./src")
import asyncio
from face_detection import face_detection
from head_pose_estimation import head_pose_estimation
from facial_landmarks_detection import facial_landmarks_detection
from gaze_estimation import gaze_estimation
from input_feeder import InputFeeder
from mouse_controller import MouseController
import numpy as np
from argparse import ArgumentParser
import time
import cv2
import math

def setup(args):
    fdmodel = face_detection(args.facedetection, args.device, args.prob_threshold, args.cpu_extension)
    fdmodel.load_model()
    hpemodel = head_pose_estimation(args.headpose, args.device, args.prob_threshold, args.cpu_extension, True)
    hpemodel.load_model()
    flmodel = facial_landmarks_detection(args.faciallandmark, args.device, args.prob_threshold, args.cpu_extension, True)
    flmodel.load_model()
    gemodel = gaze_estimation(args.gazeestimation, args.device, args.prob_threshold, args.cpu_extension)
    gemodel.load_model()
    
    feeder = InputFeeder('cam' if args.input == 'cam' else ('video' if not args.input.split('.')[-1].lower() in ['png', 'jpg', 'gif', 'tiff', 'bmp'] else 'image'), args.input)
    feeder.load_data()
    stream = feeder.next_batch()
    controller = MouseController('high', args.mouse_speed)
    return fdmodel, hpemodel, flmodel, gemodel, controller, stream
def build_argparser():
    parser = ArgumentParser()
    parser.add_argument("-fd", "--facedetection", required=True, type=str,
                        help="Specify model location for Face Detection model.")
    parser.add_argument("-fl", "--faciallandmark", required=True, type=str,
                        help="Specify model location for Facial Landmark Detection model.")
    parser.add_argument("-hp", "--headpose", required=True, type=str,
                        help="Specify Specify model location for  Head Pose Estimation model.")
    parser.add_argument("-ge", "--gazeestimation", required=True, type=str,
                        help="Specify Specify model location for Gaze Estimation model.")
    parser.add_argument("-flags", "--previewFlags", required=False, nargs='+',
                        default=[],
                        help="Specify the flags from fd, fld, hp, ge like --flags fd hp fld (Seperate each flag by space)"
                             "to see the visualization of different model outputs of each frame," 
                             "fd for Face Detection, fld for Facial Landmark Detection"
                             "hp for Head Pose Estimation, ge for Gaze Estimation." )
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Specify Path to video file or enter cam for webcam")
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-prob", "--prob_threshold", required=False, type=float,
                        default=0.6,
                        help="Probability threshold for model to detect the face accurately from the video frame.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-ms", "--mouse_speed", type=str, default="fast", help="Speed of mouse movement (fast, medium, slow)")
    return parser
    
def execute_inference(fdmodel, hpemodel, flmodel, gemodel, controller, input):
    face = fdmodel.predict(input)
    if not face.any():
        return np.array([]), np.array([]), np.array([]), np.array([])
    hpemodel.predict(face)
    flmodel.predict(face)
    headpose = hpemodel.get_async_result()
    lefteye, righteye, eyecoord, croppedface = flmodel.get_async_result()
    if not lefteye.any() or not righteye.any():
        return np.array([]), croppedface, headpose, eyecoord
    coords = gemodel.predict(np.array([headpose, lefteye, righteye, eyecoord]))
    return coords, croppedface, headpose, eyecoord
    
if __name__ == '__main__':
    argsv = build_argparser().parse_args()
    result = setup(argsv)
    stream = list(result)[-1]
    controller = list(result)[-2]
    input = next(stream, [])
    start_inference_time=time.time()
    frames = 0
    while input.any():
        coords, face, headpose, eyecoord = execute_inference(*tuple(list(result)[:-1] + [input]))
        if frames%3 == 0 and coords.any():
            asyncio.run(controller.move(coords[0], coords[1]))
        input = next(stream, [])
        frames += 1
        if argsv.previewFlags:
            previewCanvas = np.zeros((face.shape[0],face.shape[1],3), np.uint8)
            if 'fd' in argsv.previewFlags:
                previewCanvas += face
            if 'fl' in argsv.previewFlags:
                previewCanvas = cv2.rectangle(previewCanvas, (int(max(0, face.shape[1]*eyecoord[0][0][0]-5)), int(max(0, face.shape[0]*eyecoord[0][1][0]-5))), (int(face.shape[1]*eyecoord[0][0][0]+5), int(face.shape[0]*eyecoord[0][1][0]+5)), (255,255,255))
                previewCanvas = cv2.rectangle(previewCanvas, (int(max(0, face.shape[1]*eyecoord[1][0][0]-5)), int(max(0, face.shape[0]*eyecoord[1][1][0]-5))), (int(face.shape[1]*eyecoord[1][0][0]+5), int(face.shape[0]*eyecoord[1][1][0]+5)), (255,255,255))
            if 'hp' in argsv.previewFlags:
                roll = headpose[0][2]
                cv2.line(previewCanvas, (face.shape[1]*eyecoord[0][0][0], face.shape[0]*eyecoord[0][1][0]), (face.shape[1]*eyecoord[0][0][0] + 3*math.cos(roll * math.pi / 180.0), face.shape[0]*eyecoord[0][1][0] + 3*math.sin(roll * math.pi / 180.0)), (255,0,0))
                cv2.line(previewCanvas, (face.shape[1]*eyecoord[1][0][0], face.shape[0]*eyecoord[1][1][0]), (face.shape[1]*eyecoord[1][0][0] + 3*math.cos(roll * math.pi / 180.0), face.shape[0]*eyecoord[1][1][0] + 3*math.sin(roll * math.pi / 180.0)), (255,0,0))
            previewCanvas = cv2.resize(previewCanvas.astype('uint8'), (500,500))
            if 'ge' in argsv.previewFlags:
                cv2.putText(previewCanvas, str(coords), (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))
            
            cv2.imshow("visualization", previewCanvas)
            cv2.waitKey(60)
    total_time=time.time()-start_inference_time
    print(total_time)
    print('Fps:', frames/total_time)