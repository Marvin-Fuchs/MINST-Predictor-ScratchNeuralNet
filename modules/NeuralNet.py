import numpy
import scipy.special

# neural network class definition
class NeuralNet:
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        self.inputnodes = inputnodes
        self.outputnodes = outputnodes
        self.iterations=0 #gets later updated ig more than one hidden layer -> iterations of matrix calculations
        # learningrate
        self.lr = learningrate
        # activation functions (sigmoid)
        self.activation_function = lambda x: scipy.special.expit(x)
        self.inverse_activation_function = lambda x: scipy.special.logit(x)

        #additional hidden layers in ():         wih1(, wh1h2,...,whn-2hn-1,whn-1hn,) whn
        if type(hiddennodes)==list and len(hiddennodes)>1:
                #multiple hiddenlayers require to care about the first hiddenlayer and last hiddenlayer inbetween the input and output
                self.multiple_hiddenlayer = hiddennodes
                # matrices containing the wheights
                self.wih = numpy.random.normal(0.0, pow(self.inputnodes, -0.5), (self.multiple_hiddenlayer[0], self.inputnodes))
                self.who = numpy.random.normal(0.0, pow(self.multiple_hiddenlayer[-1], -0.5), (self.outputnodes, self.multiple_hiddenlayer[-1]))
                # biases
                self.bias_ih = numpy.random.normal(0.0, pow(self.inputnodes, -0.5), (self.multiple_hiddenlayer[0], 1))
                self.bias_ho = numpy.random.normal(0.0, pow(self.multiple_hiddenlayer[-1], -0.5), (self.outputnodes, 1))

                i=0
                number_of_extra_matrices=len(self.multiple_hiddenlayer)-1
                self.iterations=number_of_extra_matrices #iterations of matrix calculations from wh1h2 until whn-1hn
                self.whh_additionalLayers = []
                self.bhh_additionalLayers = []
                while i<number_of_extra_matrices:
                    number_of_columns=self.multiple_hiddenlayer[i]
                    number_of_rows=self.multiple_hiddenlayer[i+1]
                    self.whh_additionalLayers.append(numpy.random.normal(0.0, pow(number_of_columns, -0.5), (number_of_rows, number_of_columns)))
                    self.bhh_additionalLayers.append(numpy.random.normal(0.0, pow(number_of_columns, -0.5), (number_of_rows, 1)))
                    i+=1
        # normal initialization (only one input-hidden and one hidden-output layers are necessary)
        else:
            if type(hiddennodes)==list:
                self.hiddennodes=hiddennodes[0]
            else:
                self.hiddennodes = hiddennodes
            # matrices containing the wheights
            self.wih = numpy.random.normal(0.0, pow(self.inputnodes, -0.5), (self.hiddennodes, self.inputnodes))
            self.who = numpy.random.normal(0.0, pow(self.hiddennodes, -0.5), (self.outputnodes, self.hiddennodes))
            # biases
            self.bias_ih = numpy.random.normal(0.0, pow(self.inputnodes, -0.5), (self.hiddennodes, 1))
            self.bias_ho = numpy.random.normal(0.0, pow(self.hiddennodes, -0.5), (self.outputnodes, 1))


    # train the neural network
    def train(self, inputs_list, targets_list):
        # convert inputs list to 2d array
        inputs = numpy.array(inputs_list, ndmin=2).T
        inputs_=inputs
        targets = numpy.array(targets_list, ndmin=2).T

        # calculate signals into hidden layer/ wheighted sum
        hidden_inputs = numpy.dot(self.wih, inputs)
        hidden_inputs += self.bias_ih
        # calculate the signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)
        hidden_outputs_= hidden_outputs

        # hidden layer
        if self.iterations != 0:
            inputs = hidden_outputs
            hidden_outputs_list = []
            hidden_outputs_list.append(inputs)
            for i in range(self.iterations):
                w_current = self.whh_additionalLayers[i]
                b_current = self.bhh_additionalLayers[i]

                current_inputs = numpy.dot(w_current, inputs)
                current_inputs += b_current
                # calculate the signals emerging from hidden layer
                current_outputs = self.activation_function(current_inputs)
                hidden_outputs_list.append(current_outputs)
                inputs = current_outputs
            hidden_outputs = current_outputs

        # calculate signals into final output layer
        final_inputs = numpy.dot(self.who, hidden_outputs)
        final_inputs += self.bias_ho
        # calculate the signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)

        #backpropagation begins
        # output layer error is the (target - actual)
        output_errors = targets - final_outputs
        # hidden layer error is the output_errors, split by weights, recombined at hidden nodes
        hidden_errors = numpy.dot(self.who.T, output_errors)
        hidden_errors_=hidden_errors
        # calculating the errors of the hidden layer
        if self.iterations != 0:
            output_errors_ = hidden_errors
            hidden_errors_list=[]
            hidden_errors_list.append(output_errors_)
            for i in range(self.iterations):
                w_current = self.whh_additionalLayers[-(i+1)]
                current_error = numpy.dot(w_current.T,output_errors_)
                hidden_errors_list.append(current_error)
                output_errors_=current_error #errors are sorted [from ouput towards input] which equals the direction of backpropagation

        # update the weights for the links between the hidden and output layers
        self.who += self.lr * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)),numpy.transpose(hidden_outputs))
        self.bias_ho += self.lr * (output_errors * final_outputs * (1.0 - final_outputs))

        # updating the hidden weigthts
        if self.iterations != 0:
            for i in range(self.iterations):
                position=-(i + 1)
                w_current = self.whh_additionalLayers[position]
                b_current = self.bhh_additionalLayers[position]
                error_current = hidden_errors_list[i]
                output_current = hidden_outputs_list[position]
                output_next = hidden_outputs_list[position-1]
                factor=error_current*output_current*(1.0-output_current)
                b_current+=self.lr * factor
                self.bhh_additionalLayers[position]=b_current
                w_current+=self.lr * numpy.dot(factor,numpy.transpose(output_next))
                self.whh_additionalLayers[position]=w_current

        # update the weights for the links between the input and hidden layers
        hidden_errors = hidden_errors_list[-1]
        hidden_outputs = hidden_outputs_list[0]
        self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)),numpy.transpose(inputs_))
        self.bias_ih += self.lr * (hidden_errors * hidden_outputs * (1.0 - hidden_outputs))

    # query the neural network
    def feedforward(self, inputs_list):
        # convert inputs list to 2d array
        inputs = numpy.array(inputs_list, ndmin=2).T

        # calculate signals into hidden layer
        hidden_inputs = numpy.dot(self.wih, inputs)
        hidden_inputs+=self.bias_ih
        # calculate the signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)

        #hidden layer
        if self.iterations!=0:
            inputs=hidden_outputs
            for i in range(self.iterations):
                w_current=self.whh_additionalLayers[i]
                b_current=self.bhh_additionalLayers[i]

                current_inputs = numpy.dot(w_current, inputs)
                current_inputs += b_current
                # calculate the signals emerging from hidden layer
                current_outputs = self.activation_function(current_inputs)
                inputs=current_outputs
            hidden_outputs=current_outputs

        # calculate signals into final output layer
        final_inputs = numpy.dot(self.who, hidden_outputs)
        final_inputs+=self.bias_ho
        # calculate the signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)

        return final_outputs


