#!/bin/sh                                                                                           

echo "The script you are running has basename `basename $0`, dirname `dirname $0`"
echo "The present working directory is `pwd`"

mydir=`dirname $0`

cd $mydir

. ./vars # contains variables to configure this training file
. $virtualenv

GPUARRAY_FORCE_CUDA_DRIVER_LOAD=True THEANO_FLAGS=mode=FAST_RUN,floatX=float32,device=$DEVICE,gpuar\
ray.preallocate=0.8 python $NEMATUS/nmt.py \
    --model $working_dir/model/model.npz \
    --datasets $datadir/$train.bpe.$SRC $datadir/$train.bpe.$TRG \
    --valid_datasets $datadir/$dev.bpe.$SRC $datadir/$dev.bpe.$TRG \
    --dictionaries $datadir/$traindict.bpe.$SRC.json $datadir/$traindict.bpe.$TRG.json \
    --external_validation_script $working_dir/validate.sh \
    --dim_word 512 \
    --dim 1024 \
    --lrate 0.0001 \
    --optimizer adam \
    --maxlen 76 \
    --batch_size 80 \
    -valid_batch_size 40 \
    --validFreq 10000 \
    --dispFreq 1000 \
    --saveFreq 30000 \
    --sampleFreq 10000 \
    --tie_decoder_embeddings \
    --layer_normalisation \
    --multisource_type $combo_strategy \
    --extra_sources $extra_train \
    --extra_source_dicts $datadir/$traindict.bpe.$SRC.json
    --extra_valid_sources $extra_dev
