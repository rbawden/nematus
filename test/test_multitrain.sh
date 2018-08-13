#!/bin/bash

# warning: this test is useful to check if training fails, and what speed you can achieve
# the toy datasets are too small to obtain useful translation results,
# and hyperparameters are chosen for speed, not for quality.
# For a setup that preprocesses and trains a larger data set,
# check https://github.com/rsennrich/wmt16-scripts/tree/master/sample

NEMATUS=/PATH/TO/tools/nematus-multisource



mkdir -p models

THEANO_FLAGS=exception_verbosity=low,compute_test_value=warn,optimizer=fast_compile,optimizer_excluding=inplace_elemwise_optimizer \
 python $NEMATUS/nematus/nmt.py \
	--model $NEMATUS/test/models/model.npz \
	--datasets $NEMATUS/test/data/corpus.en $NEMATUS/test/data/corpus.de \
	--dictionaries $NEMATUS/test/data/vocab.en.json $NEMATUS/test/data/vocab.de.json \
	--valid_datasets $NEMATUS/test/data/corpus.en $NEMATUS/test/data/corpus.de \
	--dim_word 12 \
    	--dim 024 \
    	--lrate 0.0001 \
    	--optimizer adam \
    	--maxlen 50 \
    	--batch_size 40 \
    	--valid_batch_size 40 \
    	--validFreq 10 \
    	--dispFreq 10 \
    	--saveFreq 10 \
    	--sampleFreq 100 \
    	--tie_decoder_embeddings \
    	--layer_normalisation \
    	--multisource_type "att-gate" \
    	--extra_sources $NEMATUS/test/data/corpus.sm1.en \
    	--extra_valid_sources $NEMATUS/test/data/corpus.sm1.en
