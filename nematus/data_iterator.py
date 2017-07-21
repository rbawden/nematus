import numpy

import gzip

import shuffle
from util import load_dict

def fopen(filename, mode='r'):
    if filename.endswith('.gz'):
        return gzip.open(filename, mode)
    return open(filename, mode)

class TextIterator:
    """Simple Bitext iterator."""
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
                 aux_source=None,
                 aux_source_dicts=None):

        self.multisource=False
        if aux_source is not None:
            self.multisource=True

        if shuffle_each_epoch:
            self.source_orig = source
            self.target_orig = target

            # multi-source inputs
            if self.multisource:
                self.aux_source_orig = aux_source
                self.source, self.target, self.aux_source = \
                    shuffle.main([self.source_orig, self.target_orig, self.aux_source_orig], temporary=True)
            # single source
            else:
                self.source, self.target = shuffle.main([self.source_orig, self.target_orig], temporary=True)
        else:
            self.source = fopen(source, 'r')
            self.target = fopen(target, 'r')
            # read auxiliary (multi-source input)
            if self.multisource:
                self.aux_source = fopen(aux_source, 'r')

        self.source_dicts = []
        for source_dict in source_dicts:
            self.source_dicts.append(load_dict(source_dict))
        self.target_dict = load_dict(target_dict)

        # read auxiliary dictionaries (multi-source input)
        self.aux_source_dicts = []
        if self.multisource:
            # use separate dictionaries for auxiliary context
            if aux_source_dicts:
                for aux_source_dict in aux_source_dicts:
                    self.aux_source_dicts.append(load_dict(aux_source_dict))
            else:
                # otherwise use the same dictionary as the main input
                for source_dict in source_dicts:
                    self.aux_source_dicts.append(load_dict(source_dict))

        self.batch_size = batch_size
        self.maxlen = maxlen
        self.skip_empty = skip_empty
        self.use_factor = use_factor

        self.n_words_source = n_words_source
        self.n_words_target = n_words_target
        # TODO: add separate option for auxiliary. limit the auxiliary context to the same as the main input for now
        if self.multisource:
            self.aux_n_words_source = n_words_source
        else:
            self.aux_n_words_source = 0

        if self.n_words_source > 0:
            for d in self.source_dicts:
                for key, idx in d.items():
                    if idx >= self.n_words_source:
                        del d[key]

        if self.n_words_target > 0:
                for key, idx in self.target_dict.items():
                    if idx >= self.n_words_target:
                        del self.target_dict[key]

        if self.aux_n_words_source > 0:
            for d in self.aux_source_dicts:
                for key, idx in d.items():
                    if idx >= self.aux_n_words_source:
                        del d[key]

        self.shuffle = shuffle_each_epoch
        self.sort_by_length = sort_by_length

        self.source_buffer = []
        self.target_buffer = []
        self.aux_source_buffer = []
        self.k = batch_size * maxibatch_size

        self.end_of_data = False

    def __iter__(self):
        return self

    def __len__(self):
        return sum([1 for _ in self])
    
    def reset(self):
        if self.shuffle:
            # multi-source shuffling
            if self.aux_source:
                self.source, self.target, self.aux_source = \
                    shuffle.main([self.source_orig, self.target_orig,
                                  self.aux_source_orig], temporary=True)
            else:
                self.source, self.target = shuffle.main([self.source_orig, self.target_orig], temporary=True)
        else:
            self.source.seek(0)
            self.target.seek(0)
            if self.aux_source:
                self.aux_source.seek(0)

    def next(self):
        if self.end_of_data:
            self.end_of_data = False
            self.reset()
            raise StopIteration

        source = []
        target = []
        aux_source = []

        # fill buffer, if it's empty
        assert len(self.source_buffer) == len(self.target_buffer), 'Buffer size mismatch!'
        if self.aux_source:
            assert len(self.aux_source_buffer) == len(self.source_buffer), \
                'Incorrect length of auxiliary input!'

        # filling the buffer for the first time
        if len(self.source_buffer) == 0:
            for ss in self.source:
                ss = ss.split()
                tt = self.target.readline().split()
                if self.multisource:
                    aux_ss = self.aux_source.readline().split()
                
                if self.skip_empty and (len(ss) == 0 or len(tt) == 0):
                    continue
                if len(ss) > self.maxlen or len(tt) > self.maxlen:
                    continue

                if self.aux_source:
                    # TODO: make special option for skip w/ auxiliary context
                    if self.skip_empty and (len(aux_ss) == 0):
                        continue
                    # TODO: make special option for maxlen w/ auxiliary context
                    if len(aux_ss) > self.maxlen:
                        continue

                self.source_buffer.append(ss)
                self.target_buffer.append(tt)
                if self.multisource:
                    self.aux_source_buffer.append(aux_ss)

                if len(self.source_buffer) == self.k:
                    break

            if len(self.source_buffer) == 0 or len(self.target_buffer) == 0:
                self.end_of_data = False
                self.reset()
                raise StopIteration

            # sort by target buffer
            if self.sort_by_length:
                tlen = numpy.array([len(t) for t in self.target_buffer])
                tidx = tlen.argsort()

                _sbuf = [self.source_buffer[i] for i in tidx]
                _tbuf = [self.target_buffer[i] for i in tidx]

                self.source_buffer = _sbuf
                self.target_buffer = _tbuf

                if self.multisource:
                    _aux_sbuf = [self.aux_source_buffer[i] for i in tidx]
                    self.aux_source_buffer = _aux_sbuf

            else:
                self.source_buffer.reverse()
                self.target_buffer.reverse()
                if self.multisource:
                    self.aux_source_buffer.reverse()


        try:
            # actual work here
            while True:

                # read from source file and map to word index
                try:
                    ss = self.source_buffer.pop()
                except IndexError:
                    break
                tmp = []
                for w in ss:
                    if self.use_factor:
                        w = [self.source_dicts[i][f] if f in self.source_dicts[i] else 1 for (i,f) in enumerate(w.split('|'))]
                    else:
                        w = [self.source_dicts[0][w] if w in self.source_dicts[0] else 1]
                    tmp.append(w)
                ss = tmp

                if self.multisource:
                    try:
                        ss =     self.source_buffer.pop()
                    except IndexError:
                        break
                    tmp = []
                    for w in aux_ss:
                        if self.use_factor:
                            w = [self.aux_source_dicts[i][f] if f in self.aux_source_dicts[i] else 1 for (i, f) in enumerate(w.split('|'))]
                        else:
                            w = [self.aux_source_dicts[0][w] if w in self.aux_source_dicts[0] else 1]
                        tmp.append(w)
                    aux_ss = tmp

                # read from target file and map to word index
                tt = self.target_buffer.pop()
                tt = [self.target_dict[w] if w in self.target_dict else 1 for w in tt]
                if self.n_words_target > 0:
                    tt = [w if w < self.n_words_target else 1 for w in tt]

                source.append(ss)
                target.append(tt)
                if self.multisource:
                    aux_source.append(aux_ss)

                if len(source) >= self.batch_size or \
                        len(target) >= self.batch_size or \
                        len(aux_source) >= self.batch_size:
                    break
        except IOError:
            self.end_of_data = True

        return source, target, aux_source
