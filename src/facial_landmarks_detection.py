from model import Model_X
import cv2
import numpy as np
'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''

class facial_landmarks_detection(Model_X):
    def preprocess_output(self, outputs):
        '''
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        '''
        self.input = self.input[0].transpose(1,2,0)
        leftEye = self.input[int(self.input.shape[0]*outputs[0][1][0][0])-5:int(self.input.shape[0]*outputs[0][1][0][0])+5, int(self.input.shape[1]*outputs[0][0][0][0])-5:int(self.input.shape[1]*outputs[0][0][0][0])+5]
        rightEye = self.input[int(self.input.shape[0]*outputs[0][3][0][0])-5:int(self.input.shape[0]*outputs[0][3][0][0])+5, int(self.input.shape[1]*outputs[0][2][0][0])-5:int(self.input.shape[1]*outputs[0][2][0][0])+5]
        if leftEye.any():
            leftEye = cv2.resize(leftEye, (60, 60)).transpose((2,0,1))
            leftEye = leftEye.reshape(1, *leftEye.shape)
        if rightEye.any():
            rightEye = cv2.resize(rightEye, (60, 60)).transpose((2,0,1))
            rightEye = rightEye.reshape(1, *rightEye.shape)
        return leftEye, rightEye, ((outputs[0][0], outputs[0][1]), (outputs[0][2], outputs[0][3])), self.input
