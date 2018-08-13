#!/bin/sh                                                                                           

mydir=`dirname $0`

cd $mydir
. ./vars
. $virtualenv


ref=$datadir/$ref.$TRG
modelprefix=$working_dir/model/model

[ -d $working_dir/model/tuning ] || mkdir $working_dir/model/tuning

# decode                                                                                            
GPUARRAY_FORCE_CUDA_DRIVER_LOAD=True THEANO_FLAGS=mode=FAST_RUN,floatX=float32,device=$DEVICE,on_un\
used_input=warn,gpuarray.preallocate=0.1 time python $NEMATUS/translate.py \
      -m $modelprefix.npz.dev.npz \
      -i $datadir/$dev.bpe.$SRC -o $working_dir/model/tuning/$dev.output.dev -k 5 -n -p 1 --suppress-unk


$mydir/postprocess.sh < $working_dir/model/tuning/$dev.output.dev > $working_dir/model/tuning/$dev.\
output.postprocessed.dev

## get BLEU                                                                                         
BEST=`cat ${modelprefix}.best_bleu || echo 0`
$moses_scripts/generic/multi-bleu.perl $ref < $working_dir/model/tuning/$dev.output.postprocessed.d\
ev >> ${modelprefix}.bleu_scores
BLEU=`$moses_scripts/generic/multi-bleu.perl $ref < $working_dir/model/tuning/$dev.output.postproce\
ssed.dev | cut -f 3 -d ' ' | cut -f 1 -d ','`
BETTER=`echo "$BLEU > $BEST" | bc`

echo "BLEU = $BLEU"

if [ "$BETTER" = "1" ]; then
  echo "new best; saving"
  echo $BLEU > ${modelprefix}.best_bleu
  cp ${modelprefix}.npz.dev.npz ${modelprefix}.npz.best_bleu
fi

