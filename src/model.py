from openvino.inference_engine import IENetwork, IECore
import cv2
import numpy as np
import time
'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''

class Model_X:
    '''
    Class for the Face Detection Model.
    '''
    def __init__(self, model_name, device='CPU', threshhold=0.5,extensions=None, run_async=False):
        self.model_weights = model_name + '.bin'
        self.model_structure = model_name + '.xml'
        self.device = device
        self.extensions = extensions
        self.threshhold = threshhold
        self.run_async = run_async

    def load_model(self):
        '''
        TODO: You will need to complete this method.
        This method is for loading the model to the device specified by the user.
        If your model requires any Plugins, this is where you can load them.
        '''
        try:
            core = IECore()
            self.check_model(core)
            self.model=IENetwork(self.model_structure, self.model_weights)
            self.net = core.load_network(network=self.model, device_name=self.device, num_requests=1)
            self.input_name=next(iter(self.model.inputs))
            self.input_names = self.model.inputs
            self.input_shape=self.model.inputs[self.input_name].shape
            self.output_name=next(iter(self.model.outputs))
            self.output_shape=self.model.outputs[self.output_name].shape
        except Exception as e:
            raise ValueError("Could not Initialise the network. Have you enterred the correct model path?")    

    def predict(self, input):
        '''
        TODO: You will need to complete this method.
        This method is meant for running predictions on the input image.
        '''
        if len(self.input_names) == 1:
            input_dict={self.input_name:self.preprocess_input(input)}
        else:
            input_dict={key:val for key, val in zip(self.input_names, input[:len(self.input_names)])}
            self.input = input 
        #startInferTime = time.time()
        if self.run_async:
            self.net.start_async(request_id=0,inputs=input_dict)
            return
        self.net.infer(input_dict)
        #print('Inference for', self.model_weights, 'took', time.time() - startInferTime)
        if len(self.model.outputs) == 1:
            return self.preprocess_output(self.net.requests[0].outputs[self.output_name])
        return self.preprocess_output(np.array([self.net.requests[0].outputs[output_name] for output_name in self.model.outputs]))
    def get_async_result(self):
        self.net.requests[0].wait(-1)
        if len(self.model.outputs) == 1:
            return self.preprocess_output(self.net.requests[0].outputs[self.output_name])
        return self.preprocess_output(np.array([self.net.requests[0].outputs[output_name] for output_name in self.model.outputs]))
    def check_model(self, plugin):
        network = plugin.read_network(model=self.model_structure, weights=self.model_weights)
        supported_layers = plugin.query_network(network=network, device_name=self.device)
        unsupported_layers = [l for l in network.layers.keys() if l not in supported_layers]
        
        
        if unsupported_layers and self.device=='CPU':
            print("unsupported layers found:", unsupported_layers)
            if not self.extensions==None:
                print("Adding cpu_extension")
                self.plugin.add_extension(self.extensions, self.device)
                supported_layers = plugin.query_network(network = network, device_name=self.device)
                unsupported_layers = [l for l in network.layers.keys() if l not in supported_layers]
                if unsupported_layers:
                    print("Unsupport layers after adding extension", unsupported_layers)
                    exit(1)
                print("Unsupported layers resolved after adding extensions")
            else:
                print("Please give path to cpu extension")
                exit(1)
        
    
    def get_forward_output(self, outputs):
        output_shape = self.input_shape
        #Hard coded threshhold
        qualifiedOutputs = [o for o in outputs[0][0] if o[2] > float(self.threshhold)]
        if len(qualifiedOutputs) == 0:
            return np.array([])
        output = qualifiedOutputs[0]
        #For this use case we assume only one region is detected
        return np.array([[grid[int(output[4]*output_shape[2]):int(output[6]*output_shape[2]), int(output[3]*output_shape[3]):int(output[5]*output_shape[3])] for grid in self.input[0]]])
    
    def preprocess_input(self, input):
        '''
        Before feeding the data into the model for inference,
        you might have to preprocess it. This function is where you can do that.
        '''
        self.input = input
        if input.shape != self.input_shape:
            if input.shape[0] == 1:
                input = input[0].transpose((1,2,0))
            resized = cv2.resize(input, (self.input_shape[3], self.input_shape[2])).transpose((2,0,1))
            self.input = resized.reshape(1, *resized.shape)
            return self.input
        return self.input

    def preprocess_output(self, outputs):
        '''
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        '''
        #overriden in sub class implementations
        raise NotImplementedError
