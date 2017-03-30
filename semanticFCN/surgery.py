from __future__ import division
import caffe
import numpy as np


def transplant(new_net, net, suffix=''):
    """
    Transfer weights by copying matching parameters, coercing parameters of
    incompatible shape, and dropping unmatched parameters.

    The coercion is useful to convert fully connected layers to their
    equivalent convolutional layers, since the weights are the same and only
    the shapes are different.  In particular, equivalent fully connected and
    convolution layers have shapes O x I and O x I x H x W respectively for O
    outputs channels, I input channels, H kernel height, and W kernel width.

    Both  `net` to `new_net` arguments must be instantiated `caffe.Net`s.
    """
    for p in net.params:
        p_new = p + suffix
        if p_new not in new_net.params:
            print 'dropping', p
            continue
        for i in range(len(net.params[p])):
            if i > (len(new_net.params[p_new]) - 1):
                print 'dropping', p, i
                break
            if net.params[p][i].data.shape != new_net.params[p_new][i].data.shape:
                print 'coercing', p, i, 'from', net.params[p][i].data.shape, 'to', new_net.params[p_new][i].data.shape
            else:
                print 'copying', p, ' -> ', p_new, i
            new_net.params[p_new][i].data.flat = net.params[p][i].data.flat


def upsample_filt(size):
    """
    Make a 2D bilinear kernel suitable for upsampling of the given (h, w) size.
    """
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    return (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)


def interp(net, layers):
    """
    Set weights of each layer in layers to bilinear kernels for interpolation.
    """
    for l in layers:
        m, k, h, w = net.params[l][0].data.shape
        if m != k and k != 1:
            print 'input + output channels need to be the same or |output| == 1'
            raise
        if h != w:
            print 'filters need to be square'
            raise
        filt = upsample_filt(h)
        net.params[l][0].data[range(m), range(k), :, :] = filt


def expand_score(new_net, new_layer, net, layer):
    """
    Transplant an old score layer's parameters, with k < k' classes, into a new
    score layer with k classes s.t. the first k' are the old classes.
    """
    old_cl = net.params[layer][0].num
    new_net.params[new_layer][0].data[:old_cl][...] = net.params[layer][0].data
    new_net.params[new_layer][1].data[0,0,0,:old_cl][...] = net.params[layer][1].data


# This Function concatenates the FCN-8s-atonce with the VGG classifier by merging and adjusting the weights
def merge_caffe_models(base_prototxt, base_model, top_prototxt, top_model, stacked_prototxt, stacked_model):

    base_net = caffe.Net(base_prototxt, caffe.TRAIN)
    print 'Copying trained layers from %s...' % (base_model)
    base_net.copy_from(base_model)

    top_net = caffe.Net(top_prototxt, caffe.TRAIN)
    print 'Copying trained layers from %s...' % (top_model)
    top_net.copy_from(top_model)

    # COPY VALUES FROM BASE NETWORK
    # For each of the pretrained net sides, copy the params to the corresponding layer of the combined net:
    # (the other layers are initialised using "xavier" method)
    stacked_net = caffe.Net(stacked_prototxt, caffe.TRAIN)
    stacked_net.copy_from(base_model)

    # COPY VALUES FROM TOP NETWORK
    # For each of the pretrained net sides, copy the params to the corresponding layer of the combined net:
    # (the other layers are initialised using "xavier" method)
    layer_prefix = 'feat'
    top_params = top_net.params.keys()
    stacked_params = stacked_net.params.keys()
    for pr in top_params:
        weights = top_net.params[pr][0].data[...]  # Grab the pretrained weights
        bias = top_net.params[pr][1].data[...]  # Grab the pretrained bias
        n_channels = weights.shape[1]

        if '{}_{}'.format(layer_prefix, pr) in stacked_params:
            # Insert into new combined net
            if 'conv' in pr:
                stacked_net.params['{}_{}'.format(layer_prefix, pr)][0].data[:, 0:n_channels, :, :] = weights
                stacked_net.params['{}_{}'.format(layer_prefix, pr)][1].data[...] = bias

    # SAVE VALUES OF MERGED NETWORK
    print 'Saving stacked model to %s...' % stacked_model
    stacked_net.save(stacked_model)


# This Function concatenates the FCN-8s-atonce with the VGG classifier by merging and adjusting the weights
def reduce_fcn_dimension(base_prototxt, base_model, adapted_prototxt, adapted_model):

    base_net = caffe.Net(base_prototxt, caffe.TRAIN)
    print 'Copying trained layers from %s...' % base_model
    base_net.copy_from(base_model)

    adapted_net = caffe.Net(adapted_prototxt, caffe.TRAIN)

    # COPY VALUES FROM BASE NETWORK
    # For each of the pretrained net sides, copy the params to the corresponding layer of the combined net:
    # (the other layers are initialised using "xavier" method)

    base_params = base_net.params.keys()

    for pr in base_params:
        weights = base_net.params[pr][0].data[...]  # Grab the pretrained weights
        bias = []
        try:
            bias = base_net.params[pr][1].data[...]  # Grab the pretrained bias
        except:
            pass

        if weights.ndim > 2 and weights.shape[0] == 21 and weights.shape[1] == 21:
            adapted_net.params[pr][0].data[...] = weights[0:5, 0:5, ...]
            if len(bias) != 0:
                adapted_net.params[pr][1].data[...] = bias[0:5, 0:5, ...]
        elif weights.ndim > 1 and weights.shape[0] == 21:
            adapted_net.params[pr][0].data[...] = weights[0:5, ...]
            if len(bias) != 0:
                adapted_net.params[pr][1].data[...] = bias[0:5, ...]
        else:
            adapted_net.params[pr][0].data[...] = weights[...]
            if len(bias) != 0:
                adapted_net.params[pr][1].data[...] = bias[...]


    # SAVE VALUES OF MERGED NETWORK
    print 'Saving stacked model to %s...' % adapted_model
    adapted_net.save(adapted_model)