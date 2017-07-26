'''
Theano utility functions
'''

import sys
import json
import cPickle as pkl
import numpy
from collections import OrderedDict
import logging

import theano
import theano.tensor as tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

floatX = theano.config.floatX
numpy_floatX = numpy.typeDict[floatX]

# float16 warning
if floatX == 'float16':
   bad = True
   try:
       [major_v, minor_v, sub_v] = map(int, theano.version.short_version.split('.'))
       # When a version of Theano that supports float16 without bugs is released, add a check here
   except:
       pass
   if bad:
       print >> sys.stderr, "Warning: float16 may not be fully supported by the current version of Theano"

# push parameters to Theano shared variables
def zip_to_theano(params, tparams):
    for kk, vv in params.iteritems():
        tparams[kk].set_value(vv)


# pull parameters from Theano shared variables
def unzip_from_theano(zipped, excluding_prefix=None):
    new_params = OrderedDict()
    for kk, vv in zipped.iteritems():
        if excluding_prefix and (kk.startswith(excluding_prefix)):
            continue
        new_params[kk] = vv.get_value()
    return new_params


# get the list of parameters: Note that tparams must be OrderedDict
def itemlist(tparams):
    return [vv for kk, vv in tparams.iteritems()]

# make prefix-appended name
def pp(pp, name):
    return '%s_%s' % (pp, name)

# initialize Theano shared variables according to the initial parameters
def init_theano_params(params):
    tparams = OrderedDict()

    tparams["Wemb"] = theano.shared(params["Wemb"], name="Wemb")
    tparams["Wemb_dec"] = theano.shared(params["Wemb_dec"], name="Wemb_dec")
    tparams["encoder_b"] = theano.shared(params["encoder_b"], name="encoder_b")
    tparams["encoder_U"] = theano.shared(params["encoder_U"], name="encoder_U")
    tparams["encoder_bx"] = theano.shared(params["encoder_bx"], name="encoder_bx")
    tparams["encoder_Ux"] = theano.shared(params["encoder_Ux"], name="encoder_Ux")
    tparams["encoder_W"] = theano.shared(params["encoder_W"], name="encoder_W")
    tparams["encoder_Wx"] = theano.shared(params["encoder_Wx"], name="encoder_Wx")
    tparams["encoder_r_b"] = theano.shared(params["encoder_r_b"], name="encoder_r_b")
    tparams["encoder_r_U"] = theano.shared(params["encoder_r_U"], name="encoder_r_U")
    tparams["encoder_r_bx"] = theano.shared(params["encoder_r_bx"], name="encoder_r_bx")
    tparams["encoder_r_Ux"] = theano.shared(params["encoder_r_Ux"], name="encoder_r_Ux")
    tparams["encoder_r_W"] = theano.shared(params["encoder_r_W"], name="encoder_r_W")
    tparams["encoder_r_Wx"] = theano.shared(params["encoder_r_Wx"], name="encoder_r_Wx")
    tparams["Wemb2"] = theano.shared(params["Wemb2"], name="Wemb2")
    tparams["encoder2_b"] = theano.shared(params["encoder2_b"], name="encoder2_b")
    tparams["encoder2_U"] = theano.shared(params["encoder2_U"], name="encoder2_U")
    tparams["encoder2_bx"] = theano.shared(params["encoder2_bx"], name="encoder2_bx")
    tparams["encoder2_Ux"] = theano.shared(params["encoder2_Ux"], name="encoder2_Ux")
    tparams["encoder2_W"] = theano.shared(params["encoder2_W"], name="encoder2_W")
    tparams["encoder2_Wx"] = theano.shared(params["encoder2_Wx"], name="encoder2_Wx")
    tparams["encoder_r2_b"] = theano.shared(params["encoder_r2_b"], name="encoder_r2_b")
    tparams["encoder_r2_U"] = theano.shared(params["encoder_r2_U"], name="encoder_r2_U")
    tparams["encoder_r2_bx"] = theano.shared(params["encoder_r2_bx"], name="encoder_r2_bx")
    tparams["encoder_r2_Ux"] = theano.shared(params["encoder_r2_Ux"], name="encoder_r2_Ux")
    tparams["encoder_r2_W"] = theano.shared(params["encoder_r2_W"], name="encoder_r2_W")
    tparams["encoder_r2_Wx"] = theano.shared(params["encoder_r2_Wx"], name="encoder_r2_Wx")
    tparams["ff_state_W"] = theano.shared(params["ff_state_W"], name="ff_state_W")
    tparams["ff_state_b"] = theano.shared(params["ff_state_b"], name="ff_state_b")
    tparams["decoder_W"] = theano.shared(params["decoder_W"], name="decoder_W")
    tparams["decoder_b"] = theano.shared(params["decoder_b"], name="decoder_b")
    tparams["decoder_U"] = theano.shared(params["decoder_U"], name="decoder_U")
    tparams["decoder_Wx"] = theano.shared(params["decoder_Wx"], name="decoder_Wx")
    tparams["decoder_Ux"] = theano.shared(params["decoder_Ux"], name="decoder_Ux")
    tparams["decoder_bx"] = theano.shared(params["decoder_bx"], name="decoder_bx")
    tparams["decoder_U_nl"] = theano.shared(params["decoder_U_nl"], name="decoder_U_nl")
    tparams["decoder_b_nl"] = theano.shared(params["decoder_b_nl"], name="decoder_b_nl")
    tparams["decoder_Ux_nl"] = theano.shared(params["decoder_Ux_nl"], name="decoder_Ux_nl")
    tparams["decoder_bx_nl"] = theano.shared(params["decoder_bx_nl"], name="decoder_bx_nl")
    tparams["decoder_Wc"] = theano.shared(params["decoder_Wc"], name="decoder_Wc")
    tparams["decoder_Wcx"] = theano.shared(params["decoder_Wcx"], name="decoder_Wcx")
    tparams["decoder_W_comb_att"] = theano.shared(params["decoder_W_comb_att"], name="decoder_W_comb_att")
    tparams["decoder_Wc_att"] = theano.shared(params["decoder_Wc_att"], name="decoder_Wc_att")
    tparams["decoder_b_att"] = theano.shared(params["decoder_b_att"], name="decoder_b_att")
    tparams["decoder_U_att"] = theano.shared(params["decoder_U_att"], name="decoder_U_att")
    tparams["decoder_c_tt"] = theano.shared(params["decoder_c_tt"], name="decoder_c_tt")
    tparams["decoder_W_comb_att2"] = theano.shared(params["decoder_W_comb_att2"], name="decoder_W_comb_att2")
    tparams["decoder_Wc_att2"] = theano.shared(params["decoder_Wc_att2"], name="decoder_Wc_att2")
    tparams["decoder_b_att2"] = theano.shared(params["decoder_b_att2"], name="decoder_b_att2")
    tparams["decoder_U_att2"] = theano.shared(params["decoder_U_att2"], name="decoder_U_att2")
    tparams["decoder_c_tt2"] = theano.shared(params["decoder_c_tt2"], name="decoder_c_tt2")
    tparams["decoder_W_projcomb_att"] = theano.shared(params["decoder_W_projcomb_att"], name="decoder_W_projcomb_att")
    tparams["decoder_b_projcomb"] = theano.shared(params["decoder_b_projcomb"], name="decoder_b_projcomb")
    tparams["ff_logit_lstm_W"] = theano.shared(params["ff_logit_lstm_W"], name="ff_logit_lstm_W")
    tparams["ff_logit_lstm_b"] = theano.shared(params["ff_logit_lstm_b"], name="ff_logit_lstm_b")
    tparams["ff_logit_prev_W"] = theano.shared(params["ff_logit_prev_W"], name="ff_logit_prev_W")
    tparams["ff_logit_prev_b"] = theano.shared(params["ff_logit_prev_b"], name="ff_logit_prev_b")
    tparams["ff_logit_ctx_W"] = theano.shared(params["ff_logit_ctx_W"], name="ff_logit_ctx_W")
    tparams["ff_logit_ctx_b"] = theano.shared(params["ff_logit_ctx_b"], name="ff_logit_ctx_b")
    tparams["ff_logit_W"] = theano.shared(params["ff_logit_W"], name="ff_logit_W")
    tparams["ff_logit_b"] = theano.shared(params["ff_logit_b"], name="ff_logit_b")
    
    
    #for kk, pp in params.iteritems():
        #print('tparams['+str(kk)+'] = theano.shared(params['+str(kk)+'], name='+str(kk)+')')
    #    tparams[kk] = theano.shared(params[kk], name=kk)
    return tparams


# load parameters
def load_params(path, params, with_prefix=''):
    try:
        pp = numpy.load(path)
    except IOError:
        pp = numpy.load(path + '.npz')
    new_params = OrderedDict()
    for kk, vv in params.iteritems():
        if kk not in pp:
            logging.warn('%s is not in the archive' % kk)
            continue
        if kk == "zipped_params":
            continue
        new_params[with_prefix+kk] = pp[kk].astype(floatX, copy=False)

    params.update(new_params)
    return params

# load parameters of the optimizer
def load_optimizer_params(path, optimizer_name):
    params = {}
    try:
        pp = numpy.load(path)
    except IOError:
        pp = numpy.load(path + '.npz')
    for kk in pp:
        if kk.startswith(optimizer_name):
            params[kk] = pp[kk].astype(floatX, copy=False)
    return params

# save model parameters, optimizer parameters and progress
def save(model_params, optimizer_params, training_progress, base_filename, file_float_type='float32'):
    if file_float_type != floatX:
        new_model_params, new_optimizer_params = {}, {}
        for kk, vv in model_params.iteritems():
            new_model_params[kk] = vv.astype(file_float_type)
        for kk, vv in optimizer_params.iteritems():
            new_optimizer_params[kk] = vv.astype(file_float_type)
        model_params, optimizer_params = new_model_params, new_optimizer_params

    numpy.savez(base_filename, **model_params)
    numpy.savez(base_filename + '.gradinfo', **optimizer_params)
    training_progress.save_to_json(base_filename + '.progress.json')

def tanh(x):
    return tensor.tanh(x)


def linear(x):
    return x


def concatenate(tensor_list, axis=0):
    """
    Alternative implementation of `theano.tensor.concatenate`.
    This function does exactly the same thing, but contrary to Theano's own
    implementation, the gradient is implemented on the GPU.
    Backpropagating through `theano.tensor.concatenate` yields slowdowns
    because the inverse operation (splitting) needs to be done on the CPU.
    This implementation does not have that problem.
    :usage:
        >>> x, y = theano.tensor.matrices('x', 'y')
        >>> c = concatenate([x, y], axis=1)
    :parameters:
        - tensor_list : list
            list of Theano tensor expressions that should be concatenated.
        - axis : int
            the tensors will be joined along this axis.
    :returns:
        - out : tensor
            the concatenated tensor expression.
    """
    concat_size = sum(tt.shape[axis] for tt in tensor_list)

    output_shape = ()
    for k in range(axis):
        output_shape += (tensor_list[0].shape[k],)
    output_shape += (concat_size,)
    for k in range(axis + 1, tensor_list[0].ndim):
        output_shape += (tensor_list[0].shape[k],)

    out = tensor.zeros(output_shape)
    offset = 0
    for tt in tensor_list:
        indices = ()
        for k in range(axis):
            indices += (slice(None),)
        indices += (slice(offset, offset + tt.shape[axis]),)
        for k in range(axis + 1, tensor_list[0].ndim):
            indices += (slice(None),)

        out = tensor.set_subtensor(out[indices], tt)
        offset += tt.shape[axis]

    return out

# return name of word embedding for factor i
# special handling of factor 0 for backward compatibility
def embedding_name(i):
    if i == 0:
        return 'Wemb'
    else:
        return 'Wemb'+str(i)

# Zero out all parameters
def zero_all(params):
    for kk, vv in params.iteritems():
        vv[:] = numpy.zeros_like(vv)

