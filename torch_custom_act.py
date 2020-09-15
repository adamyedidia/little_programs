import matplotlib.pyplot as p
# Import basic libraries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from collections import OrderedDict

# Import PyTorch
import torch # import main library
from torch.autograd import Variable
import torch.nn as nn # import modules
from torch.autograd import Function # import Function to create custom activations
from torch.nn.parameter import Parameter # import Parameter to create custom activations with learnable parameters
from torch import optim # import optimizers for demonstrations
import torch.nn.functional as F # import torch functions
from torchvision import datasets, transforms # import transformations to use for demo
import pickle
 # Define a transform
transform = transforms.Compose([transforms.ToTensor()])

  # Download and load the training data for Fashion MNIST
trainset = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

# helper function to train a model
def train_model(model):
    '''
    Function trains the model and prints out the training log.
    INPUT:
        model - initialized PyTorch model ready for training.
        trainloader - PyTorch dataloader for training data.
    '''
    #setup training



    #define loss function
#    criterion = nn.NLLLoss()
#    criterion = nn.L1Loss()
    criterion = nn.CrossEntropyLoss()
    #define learning rate
    learning_rate = 0.003
    #define number of epochs
    epochs = 20
    #initialize optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    losses = []

    #run training and print out the loss to make sure that we are actually fitting to the training set
#    print('Training the model. Make sure that loss decreases after each epoch.\n')
    for e in range(epochs):
        running_loss = 0
        for images, labels in trainloader:
            images = images.view(images.shape[0], -1)
            log_ps = model(images)
            loss = criterion(log_ps, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        else:
            # print out the loss to make sure it is decreasing
            print(f"Training loss: {running_loss}")
            losses.append(running_loss)

    return losses

# simply define a silu function
def silu(input):
    '''
    Applies the Sigmoid Linear Unit (SiLU) function element-wise:
        SiLU(x) = x * sigmoid(x)
    '''
    return input * torch.sigmoid(input) # use torch.sigmoid to make sure that we created the most efficient implemetation based on builtin PyTorch functions

def decimalToUnderscore(x):
    return str(x).replace(".", "")

# create a class wrapper from PyTorch nn.Module, so
# the function now can be easily used in models
class SiLU(nn.Module):
    '''
    Applies the Sigmoid Linear Unit (SiLU) function element-wise:
        SiLU(x) = x * sigmoid(x)
    Shape:
        - Input: (N, *) where * means, any number of additional
          dimensions
        - Output: (N, *), same shape as the input
    References:
        -  Related paper:
        https://arxiv.org/pdf/1606.08415.pdf
    Examples:
        >>> m = silu()
        >>> input = torch.randn(2)
        >>> output = m(input)
    '''
    def __init__(self):
        '''
        Init method.
        '''
        super().__init__() # init the base class

    def forward(self, input):
        '''
        Forward pass of the function.
        '''
        return silu(input) # simply apply already implemented SiLU


# use SiLU with model created with Sequential

# initialize activation function
activation_function = SiLU()

# Initialize the model using nn.Sequential
model = nn.Sequential(OrderedDict([
                      ('fc1', nn.Linear(784, 256)),
                      ('activation1', activation_function), # use SiLU
                      ('fc2', nn.Linear(256, 128)),
                      ('bn2', nn.BatchNorm1d(num_features=128)),
                      ('activation2', activation_function), # use SiLU
                      ('dropout', nn.Dropout(0.3)),
                      ('fc3', nn.Linear(128, 64)),
                      ('bn3', nn.BatchNorm1d(num_features=64)),
                      ('activation3', activation_function), # use SiLU
                      ('logits', nn.Linear(64, 10)),
                      ('logsoftmax', nn.LogSoftmax(dim=1))]))

# Run training
#train_model(model)            

# create class for basic fully-connected deep neural network
class ClassifierSiLU(nn.Module):
    '''
    Demo classifier model class to demonstrate SiLU
    '''
    def __init__(self):
        super().__init__()

        # initialize layers
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 10)

    def forward(self, x):
        # make sure the input tensor is flattened
        x = x.view(x.shape[0], -1)

        # apply silu function
        x = silu(self.fc1(x))

        # apply silu function
        x = silu(self.fc2(x))
        
        # apply silu function
        x = silu(self.fc3(x))
        
        x = F.log_softmax(self.fc4(x), dim=1)

        return x

# Create demo model
model = ClassifierSiLU()
    
# Run training
#train_model(model)

class soft_exponential(nn.Module):
    '''
    Implementation of soft exponential activation.
    Shape:
        - Input: (N, *) where * means, any number of additional
          dimensions
        - Output: (N, *), same shape as the input
    Parameters:
        - alpha - trainable parameter
    References:
        - See related paper:
        https://arxiv.org/pdf/1602.01321.pdf
    Examples:
        >>> a1 = soft_exponential(256)
        >>> x = torch.randn(256)
        >>> x = a1(x)
    '''
    def __init__(self, in_features, alpha = None):
        '''
        Initialization.
        INPUT:
            - in_features: shape of the input
            - aplha: trainable parameter
            aplha is initialized with zero value by default
        '''
        super(soft_exponential,self).__init__()
        self.in_features = in_features

        # initialize alpha
        if alpha == None:
            self.alpha = Parameter(torch.tensor(0.0)) # create a tensor out of alpha
        else:
            self.alpha = Parameter(torch.tensor(alpha)) # create a tensor out of alpha
            
        self.alpha.requiresGrad = True # set requiresGrad to true!

    def forward(self, x):
        '''
        Forward pass of the function.
        Applies the function to the input elementwise.
        '''
        if (self.alpha == 0.0):
            return x

        if (self.alpha < 0.0):
            return - torch.log(1 - self.alpha * (x + self.alpha)) / self.alpha

        if (self.alpha > 0.0):
            return (torch.exp(self.alpha * x) - 1)/ self.alpha + self.alpha

# create class for basic fully-connected deep neural network
class ClassifierSExp(nn.Module):
    '''
    Basic fully-connected network to test Soft Exponential activation.
    '''
    def __init__(self):
        super().__init__()

        # initialize layers
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 10)

        # initialize Soft Exponential activation
        self.a1 = soft_exponential(256)
        self.a2 = soft_exponential(128)
        self.a3 = soft_exponential(64)

    def forward(self, x):
        # make sure the input tensor is flattened
        x = x.view(x.shape[0], -1)

        # apply Soft Exponential unit
        x = self.a1(self.fc1(x))
        x = self.a2(self.fc2(x))
        x = self.a3(self.fc3(x))
        x = F.log_softmax(self.fc4(x), dim=1)

        return x
    
model = ClassifierSExp()
#train_model(model)

class brelu(Function):
    '''
    Implementation of BReLU activation function.
    Shape:
        - Input: (N, *) where * means, any number of additional
          dimensions
        - Output: (N, *), same shape as the input
    References:
        - See BReLU paper:
        https://arxiv.org/pdf/1709.04054.pdf
    Examples:
        >>> brelu_activation = brelu.apply
        >>> t = torch.randn((5,5), dtype=torch.float, requires_grad = True)
        >>> t = brelu_activation(t)
    '''
    #both forward and backward are @staticmethods
    @staticmethod
    def forward(ctx, input):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        ctx.save_for_backward(input) # save input for backward pass

        # get lists of odd and even indices
        input_shape = input.shape[0]
        even_indices = [i for i in range(0, input_shape, 2)]
        odd_indices = [i for i in range(1, input_shape, 2)]

        # clone the input tensor
        output = input.clone()

        # apply ReLU to elements where i mod 2 == 0
        output[even_indices] = output[even_indices].clamp(min=0)

        # apply inversed ReLU to inversed elements where i mod 2 != 0
        output[odd_indices] = 0 - output[odd_indices] # reverse elements with odd indices
        output[odd_indices] = - output[odd_indices].clamp(min = 0) # apply reversed ReLU

        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        grad_input = None # set output to None

        input, = ctx.saved_tensors # restore input from context

        # check that input requires grad
        # if not requires grad we will return None to speed up computation
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.clone()

            # get lists of odd and even indices
            input_shape = input.shape[0]
            even_indices = [i for i in range(0, input_shape, 2)]
            odd_indices = [i for i in range(1, input_shape, 2)]

            # set grad_input for even_indices
            grad_input[even_indices] = (input[even_indices] >= 0).float() * grad_input[even_indices]

            # set grad_input for odd_indices
            grad_input[odd_indices] = (input[odd_indices] < 0).float() * grad_input[odd_indices]

        return grad_input

class ScaledSigmoid(Function):
    '''
    Implementation of BReLU activation function.
    Shape:
        - Input: (N, *) where * means, any number of additional
          dimensions
        - Output: (N, *), same shape as the input
    References:
        - See BReLU paper:
        https://arxiv.org/pdf/1709.04054.pdf
    Examples:
        >>> brelu_activation = brelu.apply
        >>> t = torch.randn((5,5), dtype=torch.float, requires_grad = True)
        >>> t = brelu_activation(t)
    '''
    #both forward and backward are @staticmethods

    def setK(self, k):
        self.k1 = 8
        self.k2 = 1

    @staticmethod
    def forward(ctx, input):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """

        output = torch.nn.functional.sigmoid(self.k1*input)
        ctx.save_for_backward(output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """


        return grad_input


class Threshold(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        _slope = 1
        return 1.0/(1.0+torch.exp(- _slope * x))

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_variables
        _slope = 1
        grad_input = _slope * torch.exp(- _slope * x) / torch.pow((1 + torch.exp(- _slope * x)), 2)
        return grad_input * grad_output

train=False
if train:

    for k1 in [0.0625, 0.125, 0.25]:
        for k2 in [0.0625, 0.125, 0.25, 0.5, 1]:
            print(k1, k2)

            class AsymThreshold(torch.autograd.Function):
                @staticmethod
                def forward(ctx, x):
                    output = torch.sigmoid(k1 * x)
                    outputK = torch.sigmoid(k2 * x)
                    ctx.save_for_backward(outputK)
                    return output

                @staticmethod
                def backward(ctx, grad_output):
                    outputK, = ctx.saved_variables
                    return k2 * grad_output * (1. - outputK) * outputK

            class AsymRelu(torch.autograd.Function):
                @staticmethod
                def forward(ctx, x):
                    output = torch.softplus(k1 * x)/k1
                    k2x = k2*x
                    ctx.save_for_backward(k2x)
                    return output

                @staticmethod
                def backward(ctx, grad_output):
                    k2x, = ctx.saved_variables
                    return torch.exp(k2x - torch.logaddexp(0, k2x))

            class AsymSoftmaxGood(torch.autograd.Function):

                @staticmethod
                def forward(ctx, x):
                    output = F.softmax(k1 * x, dim=1)
                    outputK = F.softmax(k2 * x, dim=1)
                    ctx.save_for_backward(outputK)
                    return output

                @staticmethod
                def backward(ctx, grad_output):
                    outputK, = ctx.saved_variables

#                    print(grad_output.shape)
                    tensor1 = torch.einsum('ij,ik->ijk', outputK, outputK)  # (m, n, n)
                    # Second we need to create an (n,n) identity of the feature vector

#                    print(tensor1.shape)
                    # ( p1  0  0  ...  )
                    # ( 0   p2 0  ...  )
                    # ( ...            )
                    tensor2 = torch.einsum('ij,jk->ijk', outputK, torch.eye(tensor1.shape[1]))  # (m, n, n)
                    # Then we need to subtract the first tensor from the second
                    # ( p1 - p1^2   -p1*p2   -p1*p3  ... )
                    # ( -p1*p2     p2 - p2^2   -p2*p3 ...)
                    # ( ...                              )
                    dSoftmax = tensor2 - tensor1
                    # Finally, we multiply the dSoftmax (da/dz) by da (dL/da) to get the gradient w.r.t. Z
                    dz = torch.einsum('ijk,ik->ij', dSoftmax, grad_output)  # (m, n)


                    return k2 * dz
#                    print(outputK)
#                    return grad_output * k2 * (1. - outputK) * outputK
#                    return grad_output * k2 * - outputK * outputK

            class AsymSoftmax(torch.autograd.Function):

                @staticmethod
                def forward(ctx, x):
                    output = F.softmax(x, dim=1)
                    outputK = F.softmax(x, dim=1)
                    ctx.save_for_backward(outputK)
                    return output

                @staticmethod
                def backward(ctx, grad_output):
                    outputK, = ctx.saved_variables
#                    print(outputK)
                    return grad_output * (1. - outputK) * outputK
#                    return grad_output * k2 * - outputK * outputK

            class AsymLogSoftmax(torch.autograd.Function):

                @staticmethod
                def forward(ctx, x):
                    output = F.log_softmax(k1 * x, dim=1)
                    outputLog = F.log_softmax(k2 * x, dim=1)
                    outputSoftmax = F.softmax(k2 * x, dim=1)
                    ctx.save_for_backward(outputSoftmax)
                    return output

                @staticmethod
                def backward(ctx, grad_output):
                    outputSoftmax, = ctx.saved_variables
                    print((1. - outputSoftmax))
                    return grad_output * k2 * (1. - outputSoftmax)# * outputSoftmax

            class AsymLogSoftmaxGood(torch.autograd.Function):

                @staticmethod
                def forward(ctx, x):
                    output = F.log_softmax(k1 * x, dim=1)
                    outputK = F.log_softmax(k2 * x, dim=1)
                    ctx.save_for_backward(outputK)
                    return output

                @staticmethod
                def backward(ctx, grad_output):
                    outputK, = ctx.saved_variables

#                    print(grad_output.shape)
                    tensor1 = torch.einsum('ij,ik->ijk', torch.ones(outputK.shape).transpose(0,1), outputK)  # (m, n, n)
                    # Second we need to create an (n,n) identity of the feature vector

#                    print(tensor1)

#                    print(tensor1.shape)
                    # ( p1  0  0  ...  )
                    # ( 0   p2 0  ...  )
                    # ( ...            )
                    tensor2 = torch.einsum('ij,jk->ijk', torch.ones(outputK.shape), torch.eye(tensor1.shape[1]))  # (m, n, n)
                    # Then we need to subtract the first tensor from the second
                    # ( p1 - p1^2   -p1*p2   -p1*p3  ... )
                    # ( -p1*p2     p2 - p2^2   -p2*p3 ...)
                    # ( ...                              )
#                    print(tensor2)
                    dSoftmax = tensor2 - tensor1
                    # Finally, we multiply the dSoftmax (da/dz) by da (dL/da) to get the gradient w.r.t. Z
                    dz = torch.einsum('ijk,ik->ij', dSoftmax, grad_output)  # (m, n)


                    return dz
#                    print(outputK)
#                    return grad_output * k2 * (1. - outputK) * outputK
#                    return grad_output * k2 * - outputK * outputK
            
            class AsymLogSoftmax2(torch.autograd.Function):

                @staticmethod
                def forward(ctx, x):
                    outputSoftmax = F.log_softmax(x, dim=1)
                    ctx.save_for_backward(outputSoftmax)
                    return outputSoftmax

                @staticmethod
                def backward(ctx, grad_output):
                    outputSoftmax, = ctx.saved_variables
                    print((1. - outputSoftmax))
                    return grad_output + (1. - outputSoftmax)# * outputSoftmax

            class ClassifierBReLU(nn.Module):
                '''
                Simple fully-connected classifier model to demonstrate BReLU activation.
                '''
                def __init__(self):
                    super(ClassifierBReLU, self).__init__()

                    # initialize layers
                    self.fc1 = nn.Linear(784, 256)
                    self.fc2 = nn.Linear(256, 128)
                    self.fc3 = nn.Linear(128, 64)
                    self.fc4 = nn.Linear(64, 10)

                    # create shortcuts for BReLU
                    self.a1 = AsymRelu.apply
                    self.a2 = AsymRelu.apply
                    self.a3 = AsymRelu.apply

                    self.ls = AsymSoftmaxGood.apply
#                    self.ls = AsymLogSoftmaxGood.apply
#                    self.ls = nn.LogSoftmax.apply
#                    self.ls = F.log_softmax.apply

                def forward(self, x):
                    # make sure the input tensor is flattened
                    x = x.view(x.shape[0], -1)

                    # apply BReLU
                    customSigmoid=False
                    if customSigmoid:
                        x = self.a1(self.fc1(x))
                        x = self.a2(self.fc2(x))
                        x = self.a3(self.fc3(x))
                    else:
                        x = torch.relu(self.fc1(x))
                        x = torch.relu(self.fc2(x))
                        x = torch.relu(self.fc3(x))                        
#                    x = F.log_softmax(self.fc4(x), dim=1)
#                    x = F.softmax(self.fc4(x), dim=1)
                    x = self.ls(self.fc4(x))
#                    x = torch.log(x)
#                    print(x)

                    return x
                

            model = ClassifierBReLU()
            losses = train_model(model)                    

#            pickle.dump(losses, open("softmax_" + decimalToUnderscore(k1) + "_" + decimalToUnderscore(k2) + ".p", "wb"))
            pickle.dump(losses, open("fixedrelusoftmax_" + decimalToUnderscore(k1) + "_" + decimalToUnderscore(k2) + ".p", "wb"))
#            pickle.dump(losses, open("softmax_control.p", "wb"))

else:

    for k1 in [0.0625, 0.125, 0.25, 0.5, 1]:
        for k2 in [0.0625]:
#    for k1 in [1,2,4,8,16]:
#        for k2 in [1,2,4,8,16]:
        #            losses = pickle.load(open("torch_" + str(k1) + "_" + str(k2) + ".p", "rb"))
#            losses = pickle.load(open("softmax_control.p", "rb"))
            losses = pickle.load(open("fixedrelusoftmax_" + decimalToUnderscore(k1) + "_" + decimalToUnderscore(k2) + ".p", "rb"))
#            losses = pickle.load(open("softmax_" + decimalToUnderscore(k1) + "_" + decimalToUnderscore(k2) + ".p", "rb"))
            print(losses)

            p.plot(losses, label=str(k1) + "_" + str(k2))
    p.legend()
    p.show()

