"""
Given a parallel corpus of sentence pairs: with one-to-one of target and source sentences,
produce the score, and optionally alignment for each pair.
"""

import sys
import argparse
import tempfile
import logging

import numpy

from data_iterator import TextIterator
from util import load_config
from alignment_util import combine_source_target_text_1to1
from compat import fill_options

from theano_util import (floatX, numpy_floatX, load_params, init_theano_params)
from nmt import (pred_probs, multi_pred_probs, build_model, build_multisource_model, prepare_data, prepare_multi_data)

from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import theano




# TODO: make generic for multi-source
def load_scorer(model, option, alignweights=None):
    # load model parameters and set theano shared variables
    param_list = numpy.load(model).files
    param_list = dict.fromkeys([key for key in param_list if not key.startswith('adam_')], 0)
    params = load_params(model, param_list)
    tparams = init_theano_params(params)

    if option['multisource_type'] is None:
        print("building single source model")
        trng, use_noise, x, x_mask, y, y_mask, opt_ret, cost = build_model(tparams, option)
        inps = [x, x_mask, y, y_mask]
    else:
        trng, use_noise, x, x_mask, aux_x, aux_x_mask, y, y_mask, opt_ret, cost = build_multisource_model(tparams, option)
        inps = [x, x_mask, aux_x, aux_x_mask, y, y_mask]

    use_noise.set_value(0.)

    if alignweights:
        logging.debug("Save weight mode ON, alignment matrix will be saved.")
        if option['multisource_type'] is not None:
            outputs = [cost, opt_ret['dec_alphas'], opt_ret['dec_alphas2']]
            f_log_probs = theano.function(inps, outputs)
        else:
            outputs = [cost, opt_ret['dec_alphas']]
            f_log_probs = theano.function(inps, outputs)
    else:
        f_log_probs = theano.function(inps, cost)

    return f_log_probs


def rescore_model(source_file, target_file, saveto, models, options, b, normalization_alpha, verbose, alignweights):
    trng = RandomStreams(1234)

    print("normalisation = "+str(normalization_alpha))

    # changed for multi-source: sources are a lsit
    def _score(pairs, alignweights=True):
        # sample given an input sequence and obtain scores
        scores = []
        sent_alignments = []
        for i, model in enumerate(models):
            f_log_probs = load_scorer(model, options[i], alignweights=alignweights)

            # TODO: make multi ?
            score, alignments = pred_probs(f_log_probs, prepare_data, options[i], pairs,
                                                  normalization_alpha=normalization_alpha, alignweights=alignweights)
            scores.append(score)
            sent_alignments.append(alignments)

        return scores, sent_alignments

    print("n words src = "+str(options[0]['n_words_src']))

    pairs = TextIterator(source_file.name, target_file.name,
                         options[0]['dictionaries'][:-1], options[0]['dictionaries'][-1],
                         n_words_source=options[0]['n_words_src'], n_words_target=options[0]['n_words'],
                         batch_size=b,
                         maxlen=float('inf'),
                         sort_by_length=False)  # TODO: sorting by length could be more efficient, but we'd want to resort after

    scores, alignments = _score(pairs, alignweights)

    source_file.seek(0)
    target_file.seek(0)
    # source_lines = source_file.readlines()
    target_lines = target_file.readlines()

    for i, line in enumerate(target_lines):
        score_str = ' '.join(map(str, [s[i] for s in scores]))
        if verbose:
            saveto.write('{0} '.format(line.strip()))
        saveto.write('{0}\n'.format(score_str))

    # optional save weights mode.
    if alignweights:
        # writing out the alignments.
        temp_name = saveto.name + ".json"
        with tempfile.NamedTemporaryFile(prefix=temp_name) as align_OUT:
            for line in alignments:
                align_OUT.write(line + "\n")
            # combining the actual source and target words.
            combine_source_target_text_1to1(source_file, target_file, saveto.name, align_OUT)


# Multi-source version of rescore model (just 2 inputs for now)
# source_files, savetos are lists
def multi_rescore_model(source_files, target_file, savetos, models, options, b,
                        normalization_alpha, verbose, alignweights, extra_sources):
    assert len(source_files) == len(savetos)  # as many inputs as different alignments

    trng = RandomStreams(1234)

    def _score(pairs, alignweights=False):
        # sample given an input sequence and obtain scores
        scores = []
        alignments = []
        aux_alignments = []
        for i, model in enumerate(models):
            f_log_probs = load_scorer(model, options[i], alignweights=alignweights)
            score, alignment, aux_alignment = multi_pred_probs(f_log_probs, prepare_multi_data, options[i],
                                                         pairs, normalization_alpha=normalization_alpha,
                                                         alignweights=alignweights)
            scores.append(score)
            alignments.append(alignment)
            aux_alignments.append(aux_alignment)

        return scores, (alignments, aux_alignments)

    # list of sources + target sentences (target sentences are the final list)
    # TODO: make TextIterator generic
    sents = TextIterator(source_files, target_file,
                         options[0]['dictionaries'][:-1], options[0]['dictionaries'][-1],
                         n_words_source=options[0]['n_words_src'], n_words_target=options[0]['n_words'],
                         batch_size=b, maxlen=float('inf'), sort_by_length=False,
                         extra_sources=extra_sources)
    # TODO: sorting by length could be more efficient, but we'd want to resort after

    scores, alignments = _score(sents, alignweights)

    source_lines = []
    extra_source_lines = []
    for ss in source_files:
        ss.seek(0)
        source_lines.append(ss.readlines())
    for xss in extra_sources:
        xss.seek(0)
        extra_source_lines.append(xss.readlines)

    target_file.seek(0)
    target_lines = target_file.readlines()

    # print out scores for each translation
    for i, line in enumerate(target_lines):
        score_str = ' '.join(map(str, [s[i] for s in scores]))
        if verbose:
            savetos[i].write('{0} '.format(line.strip()))
        savetos[i].write('{0}\n'.format(score_str))

    # optional save weights mode.
    if alignweights:
        for i, alignment in enumerate(alignments):
            # write out the alignments.
            temp_name = savetos[i].name + str(i) + ".json"
            with tempfile.NamedTemporaryFile(prefix=temp_name) as align_OUT:
                for line in alignment:
                    align_OUT.write(line + "\n")
                # combine the actual source and target words.
                combine_source_target_text_1to1(source_files[i], target_file, savetos[i].name, align_OUT, suffix=str(i))


def main(models, source_files, target_file, nbest_file, saveto, b=80, normalization_alpha=0.0, verbose=False, alignweights=False,
        extra_sources=None):
    # load model model_options
    options = []
    for model in models:
        options.append(load_config(model))

        fill_options(options[-1])

    # multi-source or single source functions
    if extra_sources is None:
        rescore_model(source_files[0], target_file, nbest_file, saveto, models, options, b, normalization_alpha, verbose, alignweights)
    else:
        savetos = [saveto+"_"+str(i) for i in range(len(source_files))]
        multi_rescore_model(source_files, target_file, nbest_file, savetos, models, options, b, normalization_alpha, verbose, alignweights,
                            extra_sources=extra_sources)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', type=int, default=80,
                        help="Minibatch size (default: %(default)s))")
    parser.add_argument('-n', type=float, default=0.0, nargs="?", const=1.0, metavar="ALPHA",
                        help="Normalize scores by sentence length (with argument, exponentiate lengths by ALPHA)")
    parser.add_argument('-v', action="store_true", help="verbose mode.")
    parser.add_argument('--models', '-m', type=str, nargs='+', required=True,
                        help="model to use. Provide multiple models (with same vocabulary) for ensemble decoding")
    parser.add_argument('--source', '-s', type=argparse.FileType('r'), required=True, metavar='PATH', nargs='+',
                        help="Source text files (first one is the main input)")
    parser.add_argument('--target', '-t', type=argparse.FileType('r'), required=True, metavar='PATH',
                        help="Target text file")
    parser.add_argument('--output', '-o', type=argparse.FileType('w'), default=sys.stdout, metavar='PATH',
                        help="Output file (default: standard output)")
    parser.add_argument('--walign', '-w', required=False, action="store_true",
                        help="Whether to store the alignment weights or not. If specified, weights will be saved in <target>.alignment")
    # added multisource arguments
    parser.add_argument('--extra_sources', nargs='+', type=argparse.FileType('r'), default=None, metavar='PATH', help="Auxiliary input file")

    args = parser.parse_args()

    # set up logging
    level = logging.DEBUG if args.v else logging.INFO
    logging.basicConfig(level=level, format='%(levelname)s: %(message)s')

    main(args.models, args.source, args.target, args.output, b=args.b, normalization_alpha=args.n, verbose=args.v,
         alignweights=args.walign, extra_sources=args.extra_sources)
