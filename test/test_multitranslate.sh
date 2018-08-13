NEMATUS=PATH_TO/nematus-multisource/nematus
SRC=de
TRG=en
modelname=~/path/to/model/model.npz
test=./data/corpus
output=OUTPUT_FILENAME
# decode                                                                                           $

python $NEMATUS/translate.py \
    -m $modelname \
    -i $test.$SRC -o $output.$TRG -k 5 -n -p 2 --suppress-unk \
    --aux_input $test.sm1.$SRC \
