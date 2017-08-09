import numpy
import gzip
import logging
import shuffle
from util import load_dict


def fopen(filename, mode='r'):
    if filename.endswith('.gz'):
        return gzip.open(filename, mode)
    return open(filename, mode)


class TextIterator:
    """Simple Bitext iterator. Extended to multiple input (text) sources"""

    def __init__(self, source, target,
                 source_dicts, target_dict,
                 batch_size=128,
                 maxlen=100,
                 n_words_source=-1,
                 n_words_target=-1,
                 skip_empty=False,
                 shuffle_each_epoch=False,
                 sort_by_length=True,
                 use_factor=False,
                 maxibatch_size=20,
                 extra_sources=[],
                 extra_source_dicts=[],      # ordered list of dictionaries for each extra input. If empty list, reuse main source dictionaries.
                 extra_source_dicts_nums=[], # number of dictionaries for each extra input (in same order as inputs)
                 extra_n_words_source=[]):   # maximum number of inputs words for each source

        # check for multiple input sources and always store as a big list of inputs
        if extra_sources is not None and len(extra_sources) > 0:
            self.multisource = True
            all_sources = [source] + extra_sources
            if len(extra_n_words_source) < 1:
                for i in range(len(extra_sources)):
                    extra_n_words_source.append(-1)
            all_n_words_sources = [n_words_source] + list(extra_n_words_source)
        else:
            self.multisource = False
            all_sources = [source]
            all_n_words_sources = [n_words_source]
        self.n_words_sources = all_n_words_sources

        # shuffle data or not
        if shuffle_each_epoch:
            self.source_orig = all_sources
            self.target_orig = target
            shuffled = shuffle.main(self.source_orig+[self.target_orig], temporary=True)
            self.all_sources, self.target = shuffled[:-1], shuffled[-1]
        else:
            self.all_sources = [fopen(ss, 'r') for ss in all_sources]
            self.target = fopen(target, 'r')

        # get source dicts
        self.all_source_dicts = [[]]
        for source_dict in source_dicts:
            self.all_source_dicts[0].append(load_dict(source_dict))

        # append extra source dicts. If none are indicated, reuse main source dictionaries
        assert len(extra_sources) == len(extra_source_dicts_nums) or len(extra_source_dicts_nums) == 0
        assert sum(extra_source_dicts_nums) == len(extra_source_dicts)
        j = 0
        for i in range(len(extra_sources)):
            extra_dicts = []
            # if number of dictionaries are specified
            if len(extra_source_dicts_nums) > i:
                for extra_dict in extra_source_dicts[j: j + extra_source_dicts_nums[i]]:
                    extra_dicts.append(load_dict(extra_dict))
                j += extra_source_dicts_nums[i]
            # otherwise just reuse main source dictionaries
            else:
                # logging.warn('No dicts provided for extra input %s so reusing main source dicts.' % str(i + 1))
                extra_dicts = self.all_source_dicts[0]
                j += len(self.all_source_dicts[0])
            self.all_source_dicts.append(extra_dicts)

        # target dict
        self.target_dict = load_dict(target_dict)

        # set other options
        self.batch_size = batch_size
        self.maxlen = maxlen
        self.skip_empty = skip_empty
        self.use_factor = use_factor

        # specify vocabulary sizes for source (list of sources) and target
        self.n_words_sources = all_n_words_sources
        self.n_words_target = n_words_target

        for i, source_dicts in enumerate(self.all_source_dicts):
            if self.n_words_sources[i] > 0:
                for d in source_dicts:
                    for key, idx in d.items():
                        if idx >= self.n_words_sources[i]:
                            del d[key]

        if self.n_words_target > 0:
            for key, idx in self.target_dict.items():
                if idx >= self.n_words_target:
                    del self.target_dict[key]

        self.shuffle = shuffle_each_epoch
        self.sort_by_length = sort_by_length

        self.all_source_buffers = [[] for _ in self.all_sources]
        self.target_buffer = []
        self.k = batch_size * maxibatch_size

        self.end_of_data = False

    def __iter__(self):
        return self

    def __len__(self):
        return sum([1 for _ in self])

    def reset(self):
        if self.shuffle:
            shuffled = shuffle.main(self.source_orig+[self.target_orig], temporary=True)
            self.all_sources, self.target = shuffled[:-1], shuffled[-1]
        else:
            [ss.seek(0) for ss in self.all_sources]
            self.target.seek(0)

    def next(self):
        if self.end_of_data:
            self.end_of_data = False
            self.reset()
            raise StopIteration

        # sources can represent multiple inputs
        sources = [[] for _ in self.all_sources]
        target = []

        # check that buffer sizes match
        assert all(len(sb) == len(self.target_buffer) for sb in self.all_source_buffers), 'Buffer size mismatch!'

        # filling the buffer for the first time
        if len(self.target_buffer) == 0:
            ss = [[] for _ in self.all_sources]

            for tt in self.target:
                tt = tt.split()
                for i, src in enumerate(self.all_sources):
                    ss[i] = src.readline().split()

                if self.skip_empty and (any(len(s) == 0 for s in ss) or len(tt) == 0):
                    continue

                if any(len(s) > self.maxlen for s in ss) or len(tt) > self.maxlen:
                    continue

                for i in range(len(self.all_sources)):
                    self.all_source_buffers[i].append(ss[i])
                self.target_buffer.append(tt)

                if any(len(sb) == self.k for sb in self.all_source_buffers):
                    break

            if any(len(sb) == 0 for sb in self.all_source_buffers) or len(self.target_buffer) == 0:
                self.end_of_data = False
                self.reset()
                raise StopIteration

            # sort by target buffer
            if self.sort_by_length:
                tlen = numpy.array([len(t) for t in self.target_buffer])
                tidx = tlen.argsort()

                _sbufs = [[sb[i] for i in tidx] for sb in self.all_source_buffers]
                _tbuf = [self.target_buffer[i] for i in tidx]

                self.all_source_buffers = _sbufs
                self.target_buffer = _tbuf

            else:
                for sb in self.all_source_buffers:
                    sb.reverse()
                self.target_buffer.reverse()

        try:
            # actual work here
            while True:

                # read from source file(s) and map to word index
                try:
                    ss = [sb.pop() for sb in self.all_source_buffers]

                except IndexError:
                    break

                for j, ss1 in enumerate(ss):
                    tmp = []
                    for w in ss1:
                        if self.use_factor:
                            w = [self.all_source_dicts[j][i][f] if f in self.all_source_dicts[j][i]
                                 else 1 for (i, f) in enumerate(w.split('|'))]
                        else:
                            w = [self.all_source_dicts[j][0][w] if w in self.all_source_dicts[j][0] else 1]
                        tmp.append(w)
                    ss[j] = tmp
                    sources[j].append(ss[j])

                # read from target file and map to word index
                tt = self.target_buffer.pop()

                tt = [self.target_dict[w] if w in self.target_dict else 1 for w in tt]

                if self.n_words_target > 0:
                    tt = [w if w < self.n_words_target else 1 for w in tt]

                target.append(tt)

                if any(len(source) >= self.batch_size for source in sources) or len(target) >= self.batch_size:
                    break
        except IOError:
            self.end_of_data = True

        return sources, target


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs=2, required=True)
    parser.add_argument("--dicts", nargs='+', required=True)
    parser.add_argument("--extra_sources", nargs='+', default=[])
    args = parser.parse_args()

    from nmt import prepare_multi_data

    ti = TextIterator(args.datasets[0], args.datasets[-1], args.dicts[:-1], args.dicts[-1],
                      extra_sources=args.extra_sources)

    for xs, y in ti:

        print("\tbefore")


        print(xs[0])


        xs, x_masks, y, y_masks = prepare_multi_data(xs, y, maxlen=50, n_factors=1)

        print("\tafter")

        if xs is not None:
            print(xs[0].shape, x_masks[0].shape)

            print(xs[0])
            print(xs[1])

            raw_input()