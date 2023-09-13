from tqdm import tqdm
import ujson as json
import numpy as np
import unidecode

vda_rel2id = {'1:NR:2': 0, '1:VDA:2': 1}

def chunks(l, n):
    res = []
    for i in range(0, len(l), n):
        assert len(l[i:i + n]) == n
        res += [l[i:i + n]]
    return res

def read_vda(file_in, tokenizer, max_seq_length=1024):
    pmids = set()
    features = []
    maxlen = 0
    with open(file_in, 'r') as infile:
        lines = infile.readlines()
        # lines = lines[:10]
        for i_l, line in enumerate(tqdm(lines)):
            line = line.rstrip().split('\t')
            pmid = line[0]

            if pmid not in pmids:
                pmids.add(pmid)
                text = line[1]
                prs = chunks(line[2:], 19)

                ent2idx = {}
                train_triples = {}

                entity_pos = set()
                for p in prs:
                    es = list(map(int, p[8].split(':')))
                    ed = list(map(int, p[9].split(':')))
                    tpy = p[7]
                    cg_name = p[-2]
                    if cg_name == 'None':
                        cg_name = 0
                    for start, end in zip(es, ed):
                        entity_pos.add((start, end, tpy, int(cg_name)))

                    es = list(map(int, p[14].split(':')))
                    ed = list(map(int, p[15].split(':')))
                    tpy = p[13]
                    cg_name = p[-1]
                    if cg_name == 'None':
                        cg_name = 0
                    for start, end in zip(es, ed):
                        entity_pos.add((start, end, tpy, cg_name))

                sents = [t.split(' ') for t in text.split('|')]
                new_sents = []
                new_cgs = []
                sent_map = {}
                i_t = 0
                for sent in sents:
                    for token in sent:
                        tokens_wordpiece = tokenizer.tokenize(token)
                        cgs_wordpiece = [0] * len(tokens_wordpiece)
                        for start, end, tpy, cg_name in list(entity_pos):
                            if i_t == start:
                                tokens_wordpiece = ["*"] + tokens_wordpiece
                                cgs_wordpiece = [cg_name] + [cg_name] * (len(tokens_wordpiece) - 1)
                            if i_t + 1 == end:
                                tokens_wordpiece = tokens_wordpiece + ["*"]
                                cgs_wordpiece = [cg_name] * (len(tokens_wordpiece) - 1) + [cg_name]
                        sent_map[i_t] = len(new_sents)
                        new_sents.extend(tokens_wordpiece)
                        new_cgs.extend(cgs_wordpiece)
                        i_t += 1
                    sent_map[i_t] = len(new_sents)
                sents = new_sents
                cgs = new_cgs

                entity_pos = []

                for p in prs:
                    if p[0] == "not_include":
                        continue
                    if p[1] == "L2R":
                        h_id, t_id = p[5], p[11]
                        h_start, t_start = p[8], p[14]
                        h_end, t_end = p[9], p[15]
                    else:
                        t_id, h_id = p[5], p[11]
                        t_start, h_start = p[8], p[14]
                        t_end, h_end = p[9], p[15]
                    h_start = map(int, h_start.split(':'))
                    h_end = map(int, h_end.split(':'))
                    t_start = map(int, t_start.split(':'))
                    t_end = map(int, t_end.split(':'))
                    h_start = [sent_map[idx] for idx in h_start]
                    h_end = [sent_map[idx] for idx in h_end]
                    t_start = [sent_map[idx] for idx in t_start]
                    t_end = [sent_map[idx] for idx in t_end]
                    if h_id not in ent2idx:
                        ent2idx[h_id] = len(ent2idx)
                        entity_pos.append(list(zip(h_start, h_end)))
                    if t_id not in ent2idx:
                        ent2idx[t_id] = len(ent2idx)
                        entity_pos.append(list(zip(t_start, t_end)))
                    h_id, t_id = ent2idx[h_id], ent2idx[t_id]

                    r = vda_rel2id[p[0]]
                    if (h_id, t_id) not in train_triples:
                        train_triples[(h_id, t_id)] = [{'relation': r}]
                    else:
                        train_triples[(h_id, t_id)].append({'relation': r})

                relations, hts = [], []
                for h, t in train_triples.keys():
                    relation = [0] * len(vda_rel2id)
                    for mention in train_triples[h, t]:
                        relation[mention["relation"]] = 1
                    relations.append(relation)
                    hts.append([h, t])

                # for h in range(len(entity_pos)):
                #     for t in range(len(entity_pos)):
                #         if (h, t) in train_triples.keys():
                #             relation = [0] * len(cdr_rel2id)
                #             for mention in train_triples[h, t]:
                #                 relation[mention["relation"]] = 1
                #                 # evidence = mention["evidence"]
                #             relations.append(relation)
                #             hts.append([h, t])
                #             # pos_samples += 1
                #         elif (h, t) not in train_triples.keys():
                #             relation = [1] + [0] * (len(cdr_rel2id) - 1)
                #             relations.append(relation)
                #             hts.append([h, t])
                #             # neg_samples += 1
                #
                # assert len(relations) == len(entity_pos) * len(entity_pos)

            maxlen = max(maxlen, len(sents))
            sents = sents[:max_seq_length - 2]
            input_ids = tokenizer.convert_tokens_to_ids(sents)
            input_ids = tokenizer.build_inputs_with_special_tokens(input_ids)
            cgs = cgs[:max_seq_length - 2]
            cg_ids = [0] + cgs + [0]

            if len(hts) > 0:
                feature = {'input_ids': input_ids,
                           'entity_pos': entity_pos,
                           'labels': relations,
                           'hts': hts,
                           'title': pmid,
                           'cg_ids': cg_ids,
                           }
                features.append(feature)
    print("Number of documents: {}.".format(len(features)))
    print("Max document length: {}.".format(maxlen))
    return features
