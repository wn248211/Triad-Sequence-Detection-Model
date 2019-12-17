# import math
# import torch.nn as nn
# import torch
#
#
# def spatial_pyramid_pool(self, previous_conv, num_sample, previous_conv_size, out_pool_size):
#     '''
#     previous_conv: a tensor vector of previous convolution layer
#     num_sample: an int number of image in the batch
#     previous_conv_size: an int vector [height, width] of the matrix features size of previous convolution layer
#     out_pool_size: a int vector of expected output size of max pooling layer
#
#     returns: a tensor vector with shape [1 x n] is the concentration of multi-level pooling
#     '''
#     print(previous_conv.size())
#     for i in range(len(out_pool_size)):
#         # print(previous_conv_size)
#         h_wid = int(math.ceil(previous_conv_size[0] / out_pool_size[i]))
#         w_wid = int(math.ceil(previous_conv_size[1] / out_pool_size[i]))
#         h_pad = (h_wid * out_pool_size[i] - previous_conv_size[0] + 1) // 2
#         w_pad = (w_wid * out_pool_size[i] - previous_conv_size[1] + 1) // 2
#
#         zero_pad = torch.nn.ZeroPad2d((w_pad, w_pad, h_pad, h_pad))
#         previous_conv = zero_pad(previous_conv)
#
#         h_new = 2 * h_pad + previous_conv_size[0]
#         w_new = 2 * w_pad + previous_conv_size[1]
#
#         kernel_size = (math.ceil(h_new / out_pool_size[i]), math.ceil(w_new / out_pool_size[i]))
#         stride = (math.floor(h_new / out_pool_size[i]), math.floor(w_new / out_pool_size[i]))
#
#         x = nn.functional.max_pool2d(previous_conv, kernel_size=kernel_size, stride=stride)
#         # x = maxpool(previous_conv)
#         if (i == 0):
#             spp = x.view(num_sample, -1)
#             # print("spp size:",spp.size())
#         else:
#             # print("size:",spp.size())
#             spp = torch.cat((spp, x.view(num_sample, -1)), 1)
#     return spp


import math
import torch.nn as nn
import torch


def spatial_pyramid_pool(self, previous_conv, num_sample, previous_conv_size, out_pool_size):
    '''
    previous_conv: a tensor vector of previous convolution layer
    num_sample: an int number of image in the batch
    previous_conv_size: an int vector [height, width] of the matrix features size of previous convolution layer
    out_pool_size: a int vector of expected output size of max pooling layer

    returns: a tensor vector with shape [1 x n] is the concentration of multi-level pooling
    '''
    # print(previous_conv.size())
    for i in range(len(out_pool_size)):
        # print(previous_conv_size)
        h_wid = int(math.ceil(previous_conv_size[0] / out_pool_size[i]))
        w_wid = int(math.ceil(previous_conv_size[1] / out_pool_size[i]))
        h_pad = (h_wid * out_pool_size[i] - previous_conv_size[0] + 1) // 2
        w_pad = (w_wid * out_pool_size[i] - previous_conv_size[1] + 1) // 2
        zero_pad = torch.nn.ZeroPad2d((w_pad, w_pad, h_pad, h_pad))
        x_new = zero_pad(previous_conv)
        h_new = 2 * h_pad + previous_conv_size[0]
        w_new = 2 * w_pad + previous_conv_size[1]
        kernel_size = (math.ceil(h_new / out_pool_size[i]), math.ceil(w_new / out_pool_size[i]))
        stride = (math.floor(h_new / out_pool_size[i]), math.floor(w_new / out_pool_size[i]))

        maxpool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride)
        x = maxpool(x_new)
        if (i == 0):
            spp = x.view(num_sample, -1)
            # print("spp size:",spp.size())
        else:
            # print("size:",spp.size())
            spp = torch.cat((spp, x.view(num_sample, -1)), 1)
    return spp