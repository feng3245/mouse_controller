from model import Model_X
import numpy as np
import math
import cv2
'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''

class gaze_estimation(Model_X):
    def preprocess_output(self, outputs):
        '''
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        '''
        roll = self.input[0][0][2]
        cosValue = math.cos(roll * math.pi / 180.0)
        sinValue = math.sin(roll * math.pi / 180.0)
        
        outputs[0][0] = outputs[0][0] * cosValue + outputs[0][1] * sinValue
        outputs[0][1] = outputs[0][0] * -sinValue + outputs[0][1] * cosValue
        return outputs[0]
