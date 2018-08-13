#!/bin/sh                                                                                           

python get_concat_last_sent.py | \
    sed -r 's/ \@(\S*?)\@ /\1/g' | \
    sed -r 's/\@\@ //g' | \
    sed "s/&lt;s&gt;//" | \
    sed "s/&apos;/'/g" | \
    /PATH/TO/tools/mosesdecoder/scripts/recaser/detruecase.perl | \
        /PATH/TO/tools/mosesdecoder/scripts/tokenizer/detokenizer.perl -l fr

