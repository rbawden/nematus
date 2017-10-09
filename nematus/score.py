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

    # compatibility with multi-source
    if 'extra_sources' not in option:
        option['extra_sources'] = []
    if 'multisource_type' not in option:
        option['multisource_type'] = None

    if 'multisource_type' not in option or option['multisource_type'] is None:
        print("building single source model")
        trng, use_noise, x, x_mask, y, y_mask, opt_ret, cost = build_model(tparams, option)
        inps = [x, x_mask, y, y_mask]
    else:
        trng, use_noise, xs, x_masks, y, y_mask, opt_ret, cost = build_multisource_model(tparams, option)
        inps = [xs[0], x_masks[0], xs[1], x_masks[1], y, y_mask]

    use_noise.set_value(0.)

    if alignweights:
        logging.debug("Save weight mode ON, alignment matrix will be saved.")
        if 'multisource_type' not in option or option['multisource_type'] is None:
            outputs = [cost, opt_ret['dec_alphas0'], opt_ret['cost_per_word']]
        else:
            outputs = [cost, opt_ret['dec_alphas0'], opt_ret['dec_alphas1'], opt_ret['cost_per_word']]
        f_log_probs = theano.function(inps, outputs)
    else:
        f_log_probs = theano.function(inps, [cost, opt_ret['cost_per_word']])

    return f_log_probs


def rescore_model(source_file, target_file, saveto, models, options, b, normalization_alpha, verbose, alignweights,
                  per_word=False):
    trng = RandomStreams(1234)

    # changed for multi-source: sources are a lsit
    def _score(pairs, alignweights=True):
        # sample given an input sequence and obtain scores
        scores = []
        sent_alignments = []
        costs_per_word = []
        for i, model in enumerate(models):
            f_log_probs = load_scorer(model, options[i], alignweights=alignweights)

            # TODO: make multi ?
            score, alignments, cost_per_word = pred_probs(f_log_probs, prepare_data, options[i], pairs,
                                                  normalization_alpha=normalization_alpha, alignweights=alignweights)

            scores.append(score)
            sent_alignments.append(alignments)
            costs_per_word.append(cost_per_word)

        return scores, sent_alignments, costs_per_word

    pairs = TextIterator(source_file.name, target_file.name,
                         options[0]['dictionaries'][:-1], options[0]['dictionaries'][-1],
                         n_words_source=options[0]['n_words_src'], n_words_target=options[0]['n_words'],
                         batch_size=b,
                         maxlen=float('inf'),
                         sort_by_length=False)  # TODO: sorting by length could be more efficient, but we'd want to resort after

    scores, alignments, costs_per_word = _score(pairs, alignweights)



    source_file.seek(0)
    target_file.seek(0)
    # source_lines = source_file.readlines()
    target_lines = target_file.readlines()

    for i, line in enumerate(target_lines):
        if per_word:
            score_str = ' '.join(map(str, [s for s in costs_per_word[0][i]][:len(line.split(" ")) + 1]))
        else:
            score_str = ' '.join(map(str, [s[i] for s in scores]))
        if verbose:
            saveto.write('{0} '.format(line.strip()))
        saveto.write('{0}\n'.format(score_str))

    # optional save weights mode.
    if alignweights:
        # writing out the alignments.
        temp_name = saveto.name + ".json"
        with tempfile.NamedTemporaryFile(prefix=temp_name) as align_OUT:
            for line in alignments[0]:
                align_OUT.write(line + "\n")
            # combining the actual source and target words.
            combine_source_target_text_1to1(source_file, target_file, saveto.name, align_OUT)


# Multi-source version of rescore model (just 2 inputs for now)
# source_files, savetos are lists
def multi_rescore_model(source_file, target_file, savetos, models, options, b,
                        normalization_alpha, verbose, alignweights, extra_sources=[], per_word=False):

    trng = RandomStreams(1234)

    def _score(pairs, alignweights=False):
        # sample given an input sequence and obtain scores
        scores = []
        #alignments = []
        #aux_alignments = []
        costs_per_word = []
        for i, model in enumerate(models):
            f_log_probs = load_scorer(model, options[i], alignweights=alignweights)
            score, all_alignments, cost_per_word = multi_pred_probs(f_log_probs, prepare_multi_data, options[i],
                                                         pairs, normalization_alpha=normalization_alpha,
                                                         alignweights=alignweights)
            scores.append(score)

            #print(all_alignments)
            #raw_input()

            #if all_alignments != []:
            #    for align in all_alignments:
            #    alignments.append(all_alignments[0])
            #    aux_alignments.append(all_alignments[1])
            costs_per_word.append(cost_per_word)

        #return scores, tuple(alignments, aux_alignments), costs_per_word
        return scores, tuple(all_alignments), costs_per_word


    print 'extra_sources', extra_sources

    # list of sources + target sentences (target sentences are the final list)
    # TODO: make TextIterator generic
    sents = TextIterator(source_file.name, target_file.name,
                         options[0]['dictionaries'][:-1], options[0]['dictionaries'][-1],
                         n_words_source=options[0]['n_words_src'], n_words_target=options[0]['n_words'],
                         batch_size=b, maxlen=float('inf'), sort_by_length=False,
                         extra_sources=[ss.name for ss in extra_sources])
    # TODO: sorting by length could be more efficient, but we'd want to resort after

    scores, all_alignments, costs_per_word = _score(sents, alignweights)

    source_lines = []
    source_file.seek(0)
    source_lines.append([source_file.readlines()])

    extra_source_lines = []
    for i, ss in enumerate(extra_sources):
        extra_sources[i].seek(0)
        extra_source_lines.append([extra_sources[i].readlines()])

    target_file.seek(0)
    target_lines = target_file.readlines()

    # print out scores for each translation
    for i, line in enumerate(target_lines):
        if per_word:
            score_str = ' '.join(map(str, [s for s in costs_per_word[0][i]][:len(line.split(" ")) + 1]))
        else:
            score_str = ' '.join(map(str, [s[i] for s in scores]))
        if verbose:
            savetos[0].write('{0} '.format(line.strip()))
        savetos[0].write('{0}\n'.format(score_str))

    # optional save weights mode.

    print 'len all alignments', len(all_alignments)
    print 'len first alignment', len(all_alignments[0])

    if alignweights:
        for i, alignments in enumerate(all_alignments):
            # write out the alignments.
            print i
            temp_name = savetos[i].name + str(i) + ".json"
            print temp_name
            with tempfile.NamedTemporaryFile(prefix=temp_name) as align_OUT:
                for line in alignments:
                    print len(line[0][0])
                    raw_input()
                    align_OUT.write(line + "\n")
                # combine the actual source and target words.
                print 'savetos', len(savetos)
                print 'source files', len(extra_sources)
                if i == 0:
                    tmp_srcfile = source_file
                else:
                    tmp_srcfile = extra_sources[i-1]
                combine_source_target_text_1to1(tmp_srcfile, target_file, savetos[i].name, align_OUT, suffix=str(i))


def main(models, source_file, target_file, saveto, b=80, normalization_alpha=0.0, verbose=False, alignweights=False,
        extra_sources=[], per_word=False):
    # load model model_options
    options = []
    for model in models:
        options.append(load_config(model))

        fill_options(options[-1])

    # multi-source or single source functions
    if len(extra_sources) == 0:
        rescore_model(source_file, target_file, saveto, models, options, b, normalization_alpha, verbose, alignweights,
                      per_word=per_word)
    else:
        savetos = [saveto] + [file(saveto.name, 'w') for _ in extra_sources]
        #source_files = source_files + extra_sources
        multi_rescore_model(source_file, target_file, savetos, models, options, b, normalization_alpha, verbose, alignweights,
                            per_word=per_word, extra_sources=extra_sources)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', type=int, default=80,
                        help="Minibatch size (default: %(default)s))")
    parser.add_argument('-n', type=float, default=0.0, nargs="?", const=1.0, metavar="ALPHA",
                        help="Normalize scores by sentence length (with argument, exponentiate lengths by ALPHA)")
    parser.add_argument('-v', action="store_true", help="verbose mode.")
    parser.add_argument('--models', '-m', type=str, nargs='+', required=True,
                        help="model to use. Provide multiple models (with same vocabulary) for ensemble decoding")
    parser.add_argument('--source', '-s', type=argparse.FileType('r'), required=True, metavar='PATH',
                        help="Source text files (first one is the main input)")
    parser.add_argument('--target', '-t', type=argparse.FileType('r'), required=True, metavar='PATH',
                        help="Target text file")
    parser.add_argument('--output', '-o', type=argparse.FileType('w'), default=sys.stdout, metavar='PATH',
                        help="Output file (default: standard output)")
    parser.add_argument('--walign', '-w', required=False, action="store_true",
                        help="Whether to store the alignment weights or not. If specified, weights will be saved in <target>.alignment")
    # added multisource arguments
    parser.add_argument('--extra_sources', nargs='+', type=argparse.FileType('r'), default=None, metavar='PATH', help="Auxiliary input file")
    # costs per word
    parser.add_argument("--per_word", default=False, action="store_true", help="Output costs per word instead of per sentence")
    args = parser.parse_args()

    # set up logging
    level = logging.DEBUG if args.v else logging.INFO
    logging.basicConfig(level=level, format='%(levelname)s: %(message)s')

    main(args.models, args.source, args.target, args.output, b=args.b, normalization_alpha=args.n, verbose=args.v,
         alignweights=args.walign, extra_sources=args.extra_sources, per_word=args.per_word)
