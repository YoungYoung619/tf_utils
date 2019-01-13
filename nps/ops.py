#-*-coding:utf-8-*-
import copy

def assign(input, position, value):
    """This function is used to assign the value to the input at the designated position.

    Args:
        input: An input array with any shape
        position: the position where you wanna assign the value, must be a list or ndarray and have
                    the corresponding dimension with input.
        value: the value which you wanna assign, a list or  ndarray which must has the same length
                with position or length 1.
    """
    def modifyValue(inputData, pos, val, k=0):
        """A recursion function which is used to change the input value at designated position
        """
        if len(inputData.shape) == 1: ## stop condition
            inputData[pos[k]] = val
        else:
            newInputData = inputData[pos[k]]
            k = k + 1
            modifyValue(newInputData, pos, val, k)

    data = copy.deepcopy(input)  ## copy a new from input
    if len(value)==1:
        for pos in position:
            modifyValue(data,pos,value[0])
    else:
        for pos,val in zip(position,value):
            modifyValue(data,pos,val)
    return data