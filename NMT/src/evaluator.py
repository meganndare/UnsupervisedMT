# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import subprocess
import re
from collections import OrderedDict
from logging import getLogger
import numpy as np
import torch
from torch import nn

from .utils import restore_segmentation


logger = getLogger()


TOOLS_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'tools')
BLEU_SCRIPT_PATH = os.path.join(TOOLS_PATH, 'mosesdecoder/scripts/generic/multi-bleu.perl')
assert os.path.isfile(BLEU_SCRIPT_PATH), "Moses not found. Please be sure you downloaded Moses in %s" % TOOLS_PATH
OG_GEN_FILE_PATH = '/localfast/mdare/code/COGS/data/gen.tsv'
PROC_GEN_FILE_PATH = '/localfast/mdare/code/unsupervised_semparse/data/gen/proc_gen.tsv'
assert os.path.isfile(OG_GEN_FILE_PATH), 'original cogs gen.tsv file not found'
assert os.path.isfile(PROC_GEN_FILE_PATH), 'preprocessed cogs gen.tsv file not found'
AMR_PROJ_ID = 'unsupervised-amr-parsing'



class EvaluatorMT(object):

    def __init__(self, trainer, data, params):
        """
        Initialize evaluator.
        """
        self.encoder = trainer.encoder
        self.decoder = trainer.decoder
        self.discriminator = trainer.discriminator
        self.data = data
        self.dico = data['dico']
        self.params = params
        self.expected_counts_by_gen_type = None
        self.src_tgt_dict = None
        self.gen_type_dict = None

        # create reference files for BLEU evaluation
        self.create_reference_files()

        # initialize dict between sentences and gen type when appropriate
        for (lang1, lang2), v in self.data['para'].items():
            if v['gen'] is not None and self.expected_counts_by_gen_type is None:
                src_tgt_dict = {}
                gen_type_dict = {}
                expected_type_counts = {}

                with open(OG_GEN_FILE_PATH, 'r', encoding='utf-8') as gens, open(PROC_GEN_FILE_PATH, 'r', encoding='utf-8') as proc_gens:
                    gen_lines = gens.readlines()
                    proc_gen_lines = proc_gens.readlines()

                    for line in proc_gen_lines:
                        src, tgt = line.split('\t')
                        src_tgt_dict[src.strip()] = tgt.strip()

                    for line in gen_lines:
                        src, tgt, type = line.split('\t')
                        if src_tgt_dict[src.strip()] in gen_type_dict:
                            print(src_tgt_dict[src.strip()])
                        gen_type_dict[src_tgt_dict[src.strip()]] = type.strip()
                        if type.strip() in expected_type_counts:
                            expected_type_counts[type.strip()] += 1
                        else:
                            expected_type_counts[type.strip()] = 1
                
                self.expected_counts_by_gen_type = expected_type_counts
                self.gen_type_dict = gen_type_dict
                self.src_tgt_dict = src_tgt_dict

    def get_pair_for_mono(self, lang):
        """
        Find a language pair for monolingual data.
        """
        candidates = [(l1, l2) for (l1, l2) in self.data['para'].keys() if l1 == lang or l2 == lang]
        assert len(candidates) > 0
        return sorted(candidates)[0]

    def mono_iterator(self, data_type, lang):
        """
        If we do not have monolingual validation / test sets, we take one from parallel data.
        """
        dataset = self.data['mono'][lang][data_type]
        if dataset is None:
            pair = self.get_pair_for_mono(lang)
            dataset = self.data['para'][pair][data_type]
            i = 0 if pair[0] == lang else 1
        else:
            i = None
        dataset.batch_size = 32
        for batch in dataset.get_iterator(shuffle=False, group_by_size=True)():
            yield batch if i is None else batch[i]

    def get_iterator(self, data_type, lang1, lang2):
        """
        Create a new iterator for a dataset.
        """
        assert data_type in ['valid', 'test', 'testtrain', 'gen']
        if lang2 is None or lang1 == lang2:
            for batch in self.mono_iterator(data_type, lang1):
                yield batch if lang2 is None else (batch, batch)
        else:
            k = (lang1, lang2) if lang1 < lang2 else (lang2, lang1)
            dataset = self.data['para'][k][data_type]
            dataset.batch_size = 32
            for batch in dataset.get_iterator(shuffle=False, group_by_size=True)():
                yield batch if lang1 < lang2 else batch[::-1]

    def create_reference_files(self):
        """
        Create reference files for BLEU evaluation.
        """
        params = self.params
        params.ref_paths = {}

        for (lang1, lang2), v in self.data['para'].items():

            assert lang1 < lang2
            lang1_id = params.lang2id[lang1]
            lang2_id = params.lang2id[lang2]

            data_types = ['valid', 'test']
            
            if v['testtrain'] is not None:
                data_types += ['testtrain']
            if v['gen'] is not None:
                data_types += ['gen']
            
            for data_type in data_types:

                lang1_path = os.path.join(params.dump_path, 'ref.{0}-{1}.{2}.txt'.format(lang2, lang1, data_type))
                lang2_path = os.path.join(params.dump_path, 'ref.{0}-{1}.{2}.txt'.format(lang1, lang2, data_type))

                lang1_txt = []
                lang2_txt = []

                # convert to text
                for (sent1, len1), (sent2, len2) in self.get_iterator(data_type, lang1, lang2):
                    lang1_txt.extend(convert_to_text(sent1, len1, self.dico[lang1], lang1_id, params))
                    lang2_txt.extend(convert_to_text(sent2, len2, self.dico[lang2], lang2_id, params))

                # replace <unk> by <<unk>> as these tokens cannot be counted in BLEU
                lang1_txt = [x.replace('<unk>', '<<unk>>') for x in lang1_txt]
                lang2_txt = [x.replace('<unk>', '<<unk>>') for x in lang2_txt]

                # export hypothesis
                with open(lang1_path, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(lang1_txt) + '\n')
                with open(lang2_path, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(lang2_txt) + '\n')

                # restore original segmentation
                restore_segmentation(lang1_path)
                restore_segmentation(lang2_path)

                # if this is AMR and we want to calculate SMATCH, create a ref file with \n\n separation
                if params.proj_name == AMR_PROJ_ID:

                    # create dirs which should not exist yet; leave pred* empty for now
                    os.makedirs(os.path.join(params.dump_path, data_type, f'ref-{lang1}-{lang2}'))
                    os.makedirs(os.path.join(params.dump_path, data_type, f'hyp-{lang1}-{lang2}'))

                    with open(lang2_path, 'r', encoding='utf-8') as ref_txt:
                        ref_lines = ref_txt.readlines()
                        for i, x in enumerate(ref_lines):
                            with open(os.path.join(params.dump_path, data_type, f'ref-{lang1}-{lang2}', f'{str(i)}.txt'), 'w', encoding='utf-8') as f:
                                f.write(x)

                # store data paths
                params.ref_paths[(lang2, lang1, data_type)] = lang1_path
                params.ref_paths[(lang1, lang2, data_type)] = lang2_path




    def eval_para(self, lang1, lang2, data_type, scores):
        """
        Evaluate lang1 -> lang2 perplexity and BLEU scores.
        """
        logger.info("Evaluating %s -> %s (%s) ..." % (lang1, lang2, data_type))
        assert data_type in ['valid', 'test', 'testtrain', 'gen']
        self.encoder.eval()
        self.decoder.eval()
        params = self.params
        lang1_id = params.lang2id[lang1]
        lang2_id = params.lang2id[lang2]

        # hypothesis
        txt = []

        # for perplexity
        loss_fn2 = nn.CrossEntropyLoss(weight=self.decoder.loss_fn[lang2_id].weight, size_average=False)
        n_words2 = self.params.n_words[lang2_id]
        count = 0
        xe_loss = 0

        for batch in self.get_iterator(data_type, lang1, lang2):

            # batch
            (sent1, len1), (sent2, len2) = batch
            sent1, sent2 = sent1.cuda(), sent2.cuda()

            # encode / decode / generate
            encoded = self.encoder(sent1, len1, lang1_id)
            decoded = self.decoder(encoded, sent2[:-1], lang2_id)
            sent2_, len2_, _ = self.decoder.generate(encoded, lang2_id)

            # cross-entropy loss
            xe_loss += loss_fn2(decoded.view(-1, n_words2), sent2[1:].view(-1)).item()
            count += (len2 - 1).sum().item()  # skip BOS word

            # convert to text
            txt.extend(convert_to_text(sent2_, len2_, self.dico[lang2], lang2_id, self.params))

        # update perplexity score
        scores['ppl_%s_%s_%s' % (lang1, lang2, data_type)] = np.exp(xe_loss / count)

        # hypothesis / reference paths
        hyp_name = 'hyp{0}.{1}-{2}.{3}.txt'.format(scores['epoch'], lang1, lang2, data_type)
        hyp_path = os.path.join(params.dump_path, hyp_name)
        ref_path = params.ref_paths[(lang1, lang2, data_type)]

        # export sentences to hypothesis file / restore BPE segmentation
        with open(hyp_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(txt) + '\n')
        restore_segmentation(hyp_path)

        # evaluate exact match accuracy
        exact_match = eval_exact_match(ref_path, hyp_path)
        logger.info("Exact match accuracy %s %s : %f" % (hyp_path, ref_path, exact_match))
        scores['em_%s_%s_%s' % (lang1, lang2, data_type)] = exact_match

        # evaluate smatch for AMR
        if params.proj_name == AMR_PROJ_ID:
            if data_type in ['valid', 'test']:
                amr_ref_path = os.path.join(params.dump_path, data_type, f'ref-{lang1}-{lang2}')
                amr_hyp_path = os.path.join(params.dump_path, data_type, f'hyp-{lang1}-{lang2}')

                # write predicted amrs
                with open(hyp_path, 'r', encoding='utf-8') as hyp_txt:
                    hyp_lines = hyp_txt.readlines()
                    for i, x in enumerate(hyp_lines):
                        with open(os.path.join(params.dump_path, data_type, f'hyp-{lang1}-{lang2}', f'{str(i)}.txt'), 'w', encoding='utf-8') as f:
                                f.write(x)
                
                avg_out_len, bucketed_avgs = eval_length_stats(ref_path, hyp_path)
                (p, r, f), well_formed_amrs = eval_smatch(amr_ref_path, amr_hyp_path)
                
                logger.info("SMATCH %s %s : (p=%f, r=%f, f=%f)" % (amr_hyp_path, amr_ref_path, p, r, f))
                logger.info("Percentage of well-formed AMRs %s %s : %f)" % (amr_hyp_path, amr_ref_path, well_formed_amrs))
                logger.info("Avg hyp output length as percent of ref %s : %f)" % (amr_hyp_path, avg_out_len))
                logger.info("Avg hyp output length as percent of ref (0-49) %s : %f)" % (amr_hyp_path, bucketed_avgs[0]))
                logger.info("Avg hyp output length as percent of ref (50-99) %s : %f)" % (amr_hyp_path, bucketed_avgs[1]))
                logger.info("Avg hyp output length as percent of ref (100-149) %s : %f)" % (amr_hyp_path, bucketed_avgs[2]))
                logger.info("Avg hyp output length as percent of ref (150-199) %s : %f)" % (amr_hyp_path, bucketed_avgs[3]))
                logger.info("Avg hyp output length as percent of ref (>200) %s : %f)" % (amr_hyp_path, bucketed_avgs[4]))
                scores['smatch_p_%s_%s_%s' % (lang1, lang2, data_type)] = p
                scores['smatch_r_%s_%s_%s' % (lang1, lang2, data_type)] = r
                scores['smatch_f_%s_%s_%s' % (lang1, lang2, data_type)] = f
                scores['wellformedness_%s_%s_%s' % (lang1, lang2, data_type)] = well_formed_amrs
                scores['len_avg_overall_%s_%s_%s' % (lang1, lang2, data_type)] = avg_out_len
                scores['len_avg_bucket_0_49_%s_%s_%s' % (lang1, lang2, data_type)] = bucketed_avgs[0]
                scores['len_avg_bucket_50_99_%s_%s_%s' % (lang1, lang2, data_type)] = bucketed_avgs[1]
                scores['len_avg_bucket_100_149_%s_%s_%s' % (lang1, lang2, data_type)] = bucketed_avgs[2]
                scores['len_avg_bucket_150_199_%s_%s_%s' % (lang1, lang2, data_type)] = bucketed_avgs[3]
                scores['len_avg_bucket_200_plus_%s_%s_%s' % (lang1, lang2, data_type)] = bucketed_avgs[4]

        # evaluate BLEU score
        if data_type in ['valid', 'test']:
            bleu = eval_moses_bleu(ref_path, hyp_path)
            logger.info("BLEU %s %s : %f" % (hyp_path, ref_path, bleu))
            scores['bleu_%s_%s_%s' % (lang1, lang2, data_type)] = bleu

        # eval generalization accuracy by type
        if data_type == 'gen':
            gen_type_accs = eval_gen_type_exact_match(ref_path, hyp_path, self.expected_counts_by_gen_type, self.gen_type_dict)
            for type, acc in gen_type_accs.items():
                logger.info("Generalization EM by type %s %s %s : %f" % (type, hyp_path, ref_path, acc))
                scores['gen_%s_%s_%s_%s' % (type, lang1, lang2, data_type)] = acc
        

    def eval_back(self, lang1, lang2, lang3, data_type, scores):
        """
        Compute lang1 -> lang2 -> lang3 perplexity and BLEU scores.
        """
        logger.info("Evaluating %s -> %s -> %s (%s) ..." % (lang1, lang2, lang3, data_type))
        assert data_type in ['valid', 'test']
        self.encoder.eval()
        self.decoder.eval()
        params = self.params
        lang1_id = params.lang2id[lang1]
        lang2_id = params.lang2id[lang2]
        lang3_id = params.lang2id[lang3]

        # hypothesis
        txt = []

        # for perplexity
        loss_fn3 = nn.CrossEntropyLoss(weight=self.decoder.loss_fn[lang3_id].weight, size_average=False)
        n_words3 = self.params.n_words[lang3_id]
        count = 0
        xe_loss = 0

        for batch in self.get_iterator(data_type, lang1, lang3):

            # batch
            (sent1, len1), (sent3, len3) = batch
            sent1, sent3 = sent1.cuda(), sent3.cuda()

            # encode / generate lang1 -> lang2
            encoded = self.encoder(sent1, len1, lang1_id)
            sent2_, len2_, _ = self.decoder.generate(encoded, lang2_id)

            # encode / decode / generate lang2 -> lang3
            encoded = self.encoder(sent2_.cuda(), len2_, lang2_id)
            decoded = self.decoder(encoded, sent3[:-1], lang3_id)
            sent3_, len3_, _ = self.decoder.generate(encoded, lang3_id)

            # cross-entropy loss
            xe_loss += loss_fn3(decoded.view(-1, n_words3), sent3[1:].view(-1)).item()
            count += (len3 - 1).sum().item()  # skip BOS word

            # convert to text
            txt.extend(convert_to_text(sent3_, len3_, self.dico[lang3], lang3_id, self.params))

        # hypothesis / reference paths
        hyp_name = 'hyp{0}.{1}-{2}-{3}.{4}.txt'.format(scores['epoch'], lang1, lang2, lang3, data_type)
        hyp_path = os.path.join(params.dump_path, hyp_name)
        if lang1 == lang3:
            _lang1, _lang3 = self.get_pair_for_mono(lang1)
            if lang3 != _lang3:
                _lang1, _lang3 = _lang3, _lang1
            ref_path = params.ref_paths[(_lang1, _lang3, data_type)]
        else:
            ref_path = params.ref_paths[(lang1, lang3, data_type)]

        # export sentences to hypothesis file / restore BPE segmentation
        with open(hyp_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(txt) + '\n')
        restore_segmentation(hyp_path)

        # evaluate BLEU score
        bleu = eval_moses_bleu(ref_path, hyp_path)
        logger.info("BLEU %s %s : %f" % (hyp_path, ref_path, bleu))

        # update scores
        scores['ppl_%s_%s_%s_%s' % (lang1, lang2, lang3, data_type)] = np.exp(xe_loss / count)
        scores['bleu_%s_%s_%s_%s' % (lang1, lang2, lang3, data_type)] = bleu

    def run_all_evals(self, epoch):
        """
        Run all evaluations.
        """
        scores = OrderedDict({'epoch': epoch})

        with torch.no_grad():

            for (lang1, lang2), v in self.data['para'].items():
                for data_type in ['gen', 'valid', 'test', 'testtrain']:
                    if v[data_type] is None:
                        continue
                    else:
                        self.eval_para(lang1, lang2, data_type, scores)
                        #self.eval_para(lang2, lang1, data_type, scores) [[[not needed for our project]]]

            for lang1, lang2, lang3 in self.params.pivo_directions:
                for data_type in ['valid', 'test']:
                    #self.eval_back(lang1, lang2, lang3, data_type, scores) [[[not needed for our project]]]
                    pass

        return scores


def eval_moses_bleu(ref, hyp):
    """
    Given a file of hypothesis and reference files,
    evaluate the BLEU score using Moses scripts.
    """
    assert os.path.isfile(ref) and os.path.isfile(hyp)
    #command = BLEU_SCRIPT_PATH + ' -r %s -t %s'
    command = BLEU_SCRIPT_PATH + ' %s < %s'
    p = subprocess.Popen(command % (ref, hyp), stdout=subprocess.PIPE, shell=True)
    result = p.communicate()[0].decode("utf-8")
    if result.startswith('BLEU'):
        return float(result[7:result.index(',')])
    else:
        logger.warning('Impossible to parse BLEU score! "%s"' % result)
        return -1
    
def eval_exact_match(ref, hyp):
    """
    Given a file of hypothesis and reference,
    evaluate the exact match accuracy
    """
    assert os.path.isfile(ref) and os.path.isfile(hyp)
    
    with open(hyp, 'r', encoding='utf-8') as pred, open(ref, 'r', encoding='utf-8') as gold:
        pred_lines = pred.readlines()
        gold_lines = gold.readlines()

        correct = 0

        assert len(pred_lines) == len(gold_lines), 'predicted sentence count does not match gold sentence count!'

        for i, line in enumerate(pred_lines):
            if line == gold_lines[i]:
                correct += 1
        
        return float(correct/len(pred_lines))
    

def eval_length_stats(ref, hyp):
    """
    Given ref and gold files, check length of predicted versus reference for each bucket
    """

    assert os.path.isfile(ref) and os.path.isfile(hyp)

    with open(hyp, 'r', encoding='utf-8') as pred, open(ref, 'r', encoding='utf-8') as gold:
        pred_lines = pred.readlines()
        gold_lines = gold.readlines()

        assert len(pred_lines) == len(gold_lines), 'predicted sentence count does not match gold sentence count!'

        output_lens_ref = np.empty(len(pred_lines))
        output_lens_hyp = np.empty(len(pred_lines))

        bucketed_len_percents = {0: [], # len 0-49
                                 1: [], # len 50-99
                                 2: [], # len 100-149
                                 3: [], # len 150-199
                                 4: [], # len > 200
                                }
        bucketed_avgs = {0:0, 
                         1:0, 
                         2:0, 
                         3:0, 
                         4:0}

        for i, line in enumerate(pred_lines):
            hyp_toks = len(line.split())
            ref_toks = len(gold_lines[i].split())
            output_lens_hyp[i] = hyp_toks
            output_lens_ref[i] = ref_toks

            if ref_toks < 50:
                bucketed_len_percents[0] += [hyp_toks/ref_toks]
            elif 50 <= ref_toks < 100:
                bucketed_len_percents[1] += [hyp_toks/ref_toks]
            elif 100 <= ref_toks < 150:
                bucketed_len_percents[2] += [hyp_toks/ref_toks]
            elif 150 <= ref_toks < 200:
                bucketed_len_percents[3] += [hyp_toks/ref_toks]
            elif ref_toks > 200:
                bucketed_len_percents[4] += [hyp_toks/ref_toks]
            else:
                raise ValueError('not a possible length of reference tokens!')

        avg_output_length = np.average(output_lens_hyp / output_lens_ref)

        logger.info(f'{avg_output_length}')
        logger.info(f'{len(bucketed_len_percents[0])}')
        logger.info(f'{len(bucketed_len_percents[1])}')
        logger.info(f'{len(bucketed_len_percents[2])}')
        logger.info(f'{len(bucketed_len_percents[3])}')
        logger.info(f'{len(bucketed_len_percents[4])}')

        for k, v in bucketed_len_percents.items():
            if len(v) > 0:
                bucketed_avgs[k] = sum(v) / len(v)
            else:
                bucketed_avgs[k] = 0.0
            logger.info(f'{bucketed_avgs[k]}')

    return avg_output_length, bucketed_avgs
    

def eval_smatch(ref, hyp):
    """
    Given a parent directory of hypothesis and reference amrs,
    evaluate the smatch score and well-formedness metric
    """
    assert os.path.isdir(ref) and os.path.isdir(hyp)
    amr_total = len(os.listdir(ref))
    assert amr_total == len(os.listdir(hyp))

    wf_count = 0
    p_scores = []
    r_scores = []
    f_scores = []

    for i in range(amr_total):

        completed_process = subprocess.run(['smatch.py', '-f', f'{hyp}/{i}.txt', f'{ref}/{i}.txt', '--pr'], timeout=1, capture_output=True)

        if completed_process.returncode == 0:
            wf_count += 1
            (p, r, f) = tuple(re.findall("\d+\.\d+", completed_process.stdout.decode("utf-8")))
            p_scores.append(p)
            r_scores.append(r)
            f_scores.append(f)

    p_scores = [float(i) for i in p_scores]
    r_scores = [float(i) for i in r_scores]
    f_scores = [float(i) for i in f_scores]

    if wf_count == 0:
        return (0.0, 0.0, 0.0), 0.0

    return (sum(p_scores) / amr_total, \
            sum(r_scores) / amr_total, \
            sum(f_scores) / amr_total), \
            wf_count / amr_total
    

def eval_gen_type_exact_match(ref, hyp, expected_counts_by_gen_type, gen_type_dict):
    assert expected_counts_by_gen_type is not None, 'expected gen type dict was never initialized!'

    correct_count_by_type = dict(zip(expected_counts_by_gen_type.keys(), [0]*len(expected_counts_by_gen_type.keys())))
    accuracy_by_type = dict(zip(expected_counts_by_gen_type.keys(), [None]*len(expected_counts_by_gen_type.keys())))

    with open(hyp, 'r', encoding='utf-8') as pred, open(ref, 'r', encoding='utf-8') as gold:
        pred_lines = pred.readlines()
        gold_lines = gold.readlines()

        correct = 0

        assert len(pred_lines) == len(gold_lines), 'predicted sentence count does not match gold sentence count!'

        for i, line in enumerate(pred_lines):
            if line == gold_lines[i]:
                correct += 1
                gen_type = gen_type_dict[gold_lines[i].strip()]
                correct_count_by_type[gen_type] += 1

    for t in correct_count_by_type:
        accuracy_by_type[t] = float(correct_count_by_type[t] / expected_counts_by_gen_type[t])
    
    return accuracy_by_type


def convert_to_text(batch, lengths, dico, lang_id, params):
    """
    Convert a batch of sentences to a list of text sentences.
    """
    batch = batch.cpu().numpy()
    lengths = lengths.cpu().numpy()
    bos_index = params.bos_index[lang_id]

    slen, bs = batch.shape
    assert lengths.max() == slen and lengths.shape[0] == bs
    assert (batch[0] == bos_index).sum() == bs
    assert (batch == params.eos_index).sum() == bs
    sentences = []

    for j in range(bs):
        words = []
        for k in range(1, lengths[j]):
            if batch[k, j] == params.eos_index:
                break
            words.append(dico[batch[k, j]])
        sentences.append(" ".join(words))
    return sentences
