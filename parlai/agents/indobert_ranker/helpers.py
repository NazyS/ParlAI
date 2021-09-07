import os
import tarfile

import parlai.core.build_data as build_data

MODEL_PATH = 'indobert-base-uncased.tar.gz'
VOCAB_PATH = 'indobert-base-uncased-vocab.txt'

MODEL_FOLDER = 'indobert_base_uncased'


def download(datapath, version='v1.0'):
    dpath = os.path.join(datapath, 'models', MODEL_FOLDER)

    if not build_data.built(dpath, version):
        print('[downloading BERT models: ' + dpath + ']')
        if build_data.built(dpath):
            # An older version exists, so remove these outdated files.
            build_data.remove_dir(dpath)
        build_data.make_dir(dpath)

        # Download the data.
        fnames = ['config.json', 'pytorch_model.bin', 'vocab.txt']
        outnames = ['bert_config.json', 'pytorch_model.bin', VOCAB_PATH]
        for fname, outname in zip(fnames, outnames):
            url = (
                'https://huggingface.co/indolem/indobert-base-uncased/resolve/main/'
                + fname
            )
            build_data.download(url, dpath, outname)

        tarfpath = os.path.join(dpath, MODEL_PATH)
        with tarfile.open(tarfpath, "w:gz") as tar:
            for file in outnames[:2]:
                fpath = os.path.join(dpath, file)
                tar.add(fpath, arcname=file)
                os.remove(fpath)

        # Mark the data as built.
        build_data.mark_done(dpath, version)
