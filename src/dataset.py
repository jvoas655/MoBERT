import torch
from word_vectorizer import WordVectorizer, POS_enumerator
from models.transformer import MotionTransformer
from torch.utils.data import Dataset, DataLoader
from os.path import join as pjoin
from tqdm import tqdm
import numpy as np
from evaluator_models import *
import os
import codecs as cs
import random
from torch.utils.data._utils.collate import default_collate
from get_opt import *
import spacy


def collate_fn3(batch):
    batch.sort(key=lambda x: x[3], reverse=True)
    return default_collate(batch)

def collate_fn2(batch):
    batch.sort(key=lambda x: x[2], reverse=True)
    return default_collate(batch)


class T2MEvalDataset(Dataset):
    def __init__(self, opt, dataset_df, motions_feats, w_vectorizer):
        self.opt = opt
        self.w_vectorizer = w_vectorizer
        self.max_motion_length = opt.max_motion_length

        self.data_dict = []
        
        nlp = spacy.load('en_core_web_sm')
        def process_text(sentence):
            sentence = sentence.replace('-', '')
            doc = nlp(sentence)
            word_list = []
            pos_list = []
            for token in doc:
                word = token.text
                if not word.isalpha():
                    continue
                if (token.pos_ == 'NOUN' or token.pos_ == 'VERB') and (word != 'left'):
                    word_list.append(token.lemma_)
                else:
                    word_list.append(word)
                pos_list.append(token.pos_)
            return word_list, pos_list
        
        for index, row in dataset_df.iterrows():
            text = row.Caption
            #tokens = text.split(" ")
            word_list, pose_list = process_text(text)
            tokens = ['%s/%s'%(word_list[i], pose_list[i]) for i in range(len(word_list))]
            motion_feats = motions_feats[row.Model][row.OriginalSample][:self.max_motion_length, ...]
            
            self.data_dict.append({
                    'motion': motion_feats,
                    'length': len(motion_feats),
                    'text': text,
                    "tokens": tokens, 
                    "dataset": row.Model,
                    "sample": row.OriginalSample,
                    "idx": index
                    })

    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, idx):
        data = self.data_dict[idx]
        motion, m_length, caption, tokens, sample = data['motion'], data['length'], data['text'], data["tokens"], data["sample"]

        if len(tokens) < self.opt.max_text_len:
            # pad with "unk"
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)
            tokens = tokens + ['unk/OTHER'] * (self.opt.max_text_len + 2 - sent_len)
        else:
            # crop
            tokens = tokens[:self.opt.max_text_len]
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)
        pos_one_hots = []
        word_embeddings = []
        for token in tokens:
            word_emb, pos_oh = self.w_vectorizer[token]
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_emb[None, :])
        pos_one_hots = np.concatenate(pos_one_hots, axis=0)
        word_embeddings = np.concatenate(word_embeddings, axis=0)

        motion = np.concatenate(
            [
                motion,
                np.zeros((self.max_motion_length - m_length, motion.shape[1]))
            ], axis=0)
        return word_embeddings, pos_one_hots, caption, len(caption.split(" ")), motion, m_length, "_".join(tokens), sample
    

'''For use of training text motion matching model, and evaluations'''
class T2MBaseDataset(Dataset):
    def __init__(self, opt, motion_data, text_data, w_vectorizer):
        self.opt = opt
        self.w_vectorizer = w_vectorizer
        self.max_motion_length = opt.max_motion_length

        self.data = []
        for sample in range(len(motion_data)):
            for sub_sample in range(len(text_data[sample])):
                m_length = len(motion_data[sample])
                motion = motion_data[sample]
                if (m_length > opt.max_motion_length):
                    motion = motion[:opt.max_motion_length, ...]
                if (m_length < 10):
                    continue
                self.data.append(
                    {
                        'motion': motion,
                        'length': len(motion),
                        "tokens": text_data[sample][sub_sample].split()
                    }
                )
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        motion, m_length, tokens = data['motion'], data['length'], data["tokens"]

        if len(tokens) < self.opt.max_text_len:
            # pad with "unk"
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)
            tokens = tokens + ['unk/OTHER'] * (self.opt.max_text_len + 2 - sent_len)
        else:
            # crop
            tokens = tokens[:self.opt.max_text_len]
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)
        pos_one_hots = []
        word_embeddings = []
        for token in tokens:
            word_emb, pos_oh = self.w_vectorizer[token]
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_emb[None, :])
        pos_one_hots = np.concatenate(pos_one_hots, axis=0)
        word_embeddings = np.concatenate(word_embeddings, axis=0)
        motion = np.concatenate(
            [
                motion,
                np.zeros((self.max_motion_length - m_length, motion.shape[1]))
            ], axis=0)
        return word_embeddings, pos_one_hots, sent_len, motion, m_length, "_".join(tokens)



def build_models(opt):
    movement_enc = MovementConvEncoder(opt.dim_pose-4, opt.dim_movement_enc_hidden, opt.dim_movement_latent)
    text_enc = TextEncoderBiGRUCo(word_size=opt.dim_word,
                                  pos_size=opt.dim_pos_ohot,
                                  hidden_size=opt.dim_text_hidden,
                                  output_size=opt.dim_coemb_hidden,
                                  device=opt.device)

    motion_enc = MotionEncoderBiGRUCo(input_size=opt.dim_movement_latent,
                                      hidden_size=opt.dim_motion_hidden,
                                      output_size=opt.dim_coemb_hidden,
                                      device=opt.device)
    print(Path(pjoin(opt.checkpoints_dir, opt.dataset_name, 'text_mot_match', 'model', 'finest.tar')))
    checkpoint = torch.load(Path(pjoin(opt.checkpoints_dir, opt.dataset_name, 'text_mot_match', 'model', 'finest.tar')),
                            map_location=opt.device)
    movement_enc.load_state_dict(checkpoint['movement_encoder'])
    text_enc.load_state_dict(checkpoint['text_encoder'])
    motion_enc.load_state_dict(checkpoint['motion_encoder'])
    print('Loading Evaluation Model Wrapper (Epoch %d) Completed!!' % (checkpoint['epoch']))
    return text_enc, motion_enc, movement_enc


class EvaluatorModelWrapper(object):

    def __init__(self, opt):

        opt.dim_pose = 263


        opt.dim_word = 300
        opt.max_motion_length = 196
        opt.dim_pos_ohot = len(POS_enumerator)
        opt.dim_motion_hidden = 1024
        opt.max_text_len = 20
        opt.dim_text_hidden = 512
        opt.dim_coemb_hidden = 512

        self.text_encoder, self.motion_encoder, self.movement_encoder = build_models(opt)
        self.opt = opt
        self.device = opt.device
        print(self.device)

        self.text_encoder.to(opt.device)
        self.motion_encoder.to(opt.device)
        self.movement_encoder.to(opt.device)

        self.text_encoder.eval()
        self.motion_encoder.eval()
        self.movement_encoder.eval()
    
    # Please note that the results does not following the order of inputs
    def get_co_embeddings(self, word_embs, pos_ohot, cap_lens, motions, m_lens):
        with torch.no_grad():
            word_embs = word_embs.detach().to(self.device).float()
            pos_ohot = pos_ohot.detach().to(self.device).float()
            motions = motions.detach().to(self.device).float()

            align_idx = np.argsort(m_lens.data.tolist())[::-1].copy()
            align_align_idx = np.argsort(align_idx).copy()
            motions = motions[align_idx]
            m_lens = m_lens[align_idx]

            '''Movement Encoding'''
            movements = self.movement_encoder(motions[..., :-4]).detach()
            m_lens = m_lens // self.opt.unit_length
            motion_embedding = self.motion_encoder(movements, m_lens)

            '''Text Encoding'''
            text_embedding = self.text_encoder(word_embs, pos_ohot, cap_lens)
            #text_embedding = text_embedding[align_idx]
        return text_embedding[align_align_idx], motion_embedding[align_align_idx]

    # Please note that the results does not following the order of inputs
    def get_motion_embeddings(self, motions, m_lens):
        with torch.no_grad():
            motions = motions.detach().to(self.device).float()

            align_idx = np.argsort(m_lens.data.tolist())[::-1].copy()
            align_align_idx = np.argsort(align_idx).copy()
            motions = motions[align_idx]
            m_lens = m_lens[align_idx]

            '''Movement Encoding'''
            movements = self.movement_encoder(motions[..., :-4]).detach()
            m_lens = m_lens // self.opt.unit_length
            motion_embedding = self.motion_encoder(movements, m_lens)
        return motion_embedding[align_align_idx]


