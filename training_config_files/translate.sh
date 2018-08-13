#!/bin/bash                                                                                         

model_dir=`dirname $0`

# language-dependent variables (source and target language)                                          
. $model_dir/vars
                                        
DEVICE=cuda0
                                                                    
cat $test.bpe.$SRC | \
     GPUARRAY_FORCE_CUDA_DRIVER_LOAD=True THEANO_FLAGS=mode=FAST_RUN,floatX=float32,device=$DEVICE \
     python $NEMATUS/translate.py \
                -m $model_dir/model/model.npz \
                -k 12 -n -p 1 --suppress-unk | $model_dir/postprocess.sh

