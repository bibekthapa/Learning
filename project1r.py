import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from datetime import datetime 

""" eta is the learning rate which is required 
during back propagation.Epochs is set to 10.
Prediction,validation_accuracyarray
are list required for the purpose of testing
and validating the results.epoch_count is required 
during plotting of the data"""

target1=[]
eta=0.2
epochs=10
prediction =[]
validation_accuracyarray=[]
epoch_time=[]
epoch_count=0

###Function to calculate the sigmoid
def sigmoid(z):
    return (1/(1+np.exp(-z)))

def errorInNetwork(target,output):
        return 0.5*(output-target)**2    

#Back Propagation Functions

"""Function to calculate error for ouput neuron"""
def errorForOutput(target,output):
    return np.multiply(output,np.multiply((1-output),(target-output)))

#Function to calculate the error for hidden layers
def errorForHidden(hidden,weight_hidden,error):
    return np.multiply(hidden,np.multiply((1-hidden),(np.dot(weight_hidden,error.T))))

"""Function to update the weight from hidden to output"""
def weight_update_wh(errorinOutput,hidden_withBias,weight_hidden):
        weight_hidden=weight_hidden+eta*np.dot(hidden_withBias,errorinOutput)
        return weight_hidden

#Function to calculate the weight from input to hidden
def weight_update_wi(errorInHidden,input_withBias,weight_input):
        weight_input=weight_input+eta*np.dot(input_withBias.T,errorInHidden.T)
        return weight_input

class HandwritingNN:

    def __init__(self,inputs):
        self.inputs = inputs
        self.l = len(self.inputs)    
        self.li = len(self.inputs[0])
        self.hidden = []
        self.hidden_neuron = 50
        self.bias = 1
        self.outputsize=10
        self.wi = np.random.random((self.li,self.hidden_neuron))*0.001
        self.wh = np.random.random((self.hidden_neuron+1,self.outputsize))*0.001
       

    def test(self,inputs):
            for i in range(len(inputs)):
                ###Reshaping of inputs so that we can feed one input at a time
                reshapedInput=np.reshape(inputs[i],(1,inputs.shape[1]))
                ###Normalization of inputs
                reshapedInput=(reshapedInput/255)
                ####Adding the bias values to the first column of the input
                inputvalues=np.asmatrix(np.insert(reshapedInput,0,self.bias))
                hidden_sigma=np.transpose(np.matrix(np.dot(inputvalues,self.wi)))
                hidden=sigmoid(hidden_sigma)
                hidden_withBias=np.insert(hidden,0,self.bias,axis=0)
                output_sigma=np.dot(hidden_withBias.T,self.wh)
                output=sigmoid(output_sigma)
                prediction.append(np.argmax(output))

    def training(self,inputs):
        for i in range(epochs):
            global epoch_count
            epoch_count=i+1
            start_time=datetime.now()
            print("Epoch {}:".format(i))

        """Input is shuffled after every epoch.Last 2000 data from 
        the train.csv is taken in validation_data whereas first 
        38000 data is taken for the training data in inputs"""

            np.random.shuffle(inputs)
            validation_data = inputs[-2000:,:]
            inputs = inputs[:38000,:]
            for i in range(len(inputs)):
                
                reshapedInput=np.reshape(inputs[i],(1,inputs.shape[1]))
                #reshapedInput = inputs
                target=reshapedInput[0][0]
                target_matrix=np.asmatrix(np.zeros(10))
                target_matrix[0,target]=1
                reshapedInput=(reshapedInput/255)
                reshapedInput[0][0]=self.bias
                hidden_sigma=np.transpose(np.matrix(np.dot(reshapedInput,self.wi)))
                hidden=sigmoid(hidden_sigma)
                hidden_withBias=np.insert(hidden,0,self.bias,axis=0)
                output_sigma=np.dot(hidden_withBias.T,self.wh)
                output=sigmoid(output_sigma)
                errorinOutput=errorForOutput(target_matrix,output)
                errorInHidden=errorForHidden(hidden,self.wh[1:],errorinOutput)
                self.wh=weight_update_wh(errorinOutput,hidden_withBias,self.wh)
                self.wi=weight_update_wi(errorInHidden,reshapedInput,self.wi)
                
            validaton_label = validation_data[:,:1].T
            validation_prediction = []
            for i in range(len(validation_data)):
                reshapedInput=np.reshape(validation_data[i],(1,validation_data.shape[1]))
                reshapedInput=(reshapedInput/255)
                reshapedInput[0][0]=self.bias
                inputvalues = reshapedInput
                hidden_sigma=np.transpose(np.matrix(np.dot(inputvalues,self.wi)))
                hidden=sigmoid(hidden_sigma)
                hidden_withBias=np.insert(hidden,0,self.bias,axis=0)
                output_sigma=np.dot(hidden_withBias.T,self.wh)
                output=sigmoid(output_sigma)
                validation_prediction.append(np.argmax(output))
                
            validation_prediction=np.asmatrix(validation_prediction)
            validation_result=np.equal(validaton_label.T,validation_prediction.T)
            validation_accuracy=(np.sum(validation_result)/np.size(validation_result,0))*100
            print ("ValiDation Accuracy : ")
            print(validation_accuracy)
            #Append validation Accuracy in a array
            validation_accuracyarray.append(validation_accuracy)
            end_time=datetime.now()
            epoch_time.append((end_time-start_time).total_seconds())
            if (validation_accuracy > 95):
                
                return

"""Taking the inputs from the file 
and processing them suitable for calculation"""

inputs = pd.read_csv("train.csv")
inputarray=np.array(inputs)
tests=pd.read_csv("test_data.csv")
testarray=np.array(tests)
labels=pd.read_csv("test_labels.csv")
testlabel=np.array(labels)

"""Object creation for the HandwritingNN class"""

a = HandwritingNN(inputarray)
a.training(inputarray)
a.test(testarray)
prediction=np.asmatrix(prediction).T
result=np.equal(testlabel,prediction)
accuracy=(np.sum(result)/np.size(result,0))*100
print ("Test Accuracy : ")
print(accuracy)


"""Plot functions of the TimeVsEpochs 
and ValidationAccuracyVsEpochs"""

plt.close('all')
x=np.arange(0,epoch_count)
f, axarr = plt.subplots(2, sharex=True)
f.subplots_adjust(wspace=2)
axarr[0].plot(x,epoch_time,"bo-")
axarr[0].set_title("Time Per Epoch (HL={},epochs={})".format(a.hidden_neuron,epochs))
axarr[0].set_ylabel("Time")
for X,Y in zip(x,epoch_time):
    axarr[0].annotate('{}'.format(np.round(Y,2)),xy=(X,Y), textcoords='offset points',xytext=(-5,5))
axarr[1].plot(x,validation_accuracyarray,"bo-")
axarr[1].set_title("Validation Accuracy with epochs(HL={},epochs={})".format(a.hidden_neuron,epochs))
axarr[1].set_ylim(0,100)
axarr[1].set_xlabel("Epochs")
axarr[1].set_ylabel("Accuracy")
for X,Y in zip(x,validation_accuracyarray):
    axarr[1].annotate('{}'.format(np.round(Y,2)),xy=(X,Y), textcoords='offset points',xytext=(-5,5))
plt.show()













