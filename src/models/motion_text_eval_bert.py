
#from collections import OrderedDict
from pathlib import Path
import pickle
#import random

#import tqdm
import numpy as np
import torch
import torch.nn as nn
#from models.motion_chunk_vae import *
import os
#from torch.nn.utils.rnn import pad_sequence

from utils.stat_tracking import *
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
from sentence_transformers import SentenceTransformer
import concurrent.futures
from tokenizers import Tokenizer, pre_tokenizers, models, trainers, processors, Encoding

class TMEvalTokenizer:
    def __init__(self, tokenization_method, training_file_path, vocab_size, saved_tokenizer_path="", use_seg_embs=True):
        self.use_seg_embs = use_seg_embs
        self.tokenization_method = tokenization_method
        self.training_file_path = training_file_path
        self.special_tokens = ["[PAD]", "[UNK]", "[MASK]", "[CLS]", "[MOT]", "[SOT]"]

        if saved_tokenizer_path != "":
            self.tokenizer = Tokenizer.from_file(str(saved_tokenizer_path))
        else:
            if self.tokenization_method == "WordPiece":
                model = models.WordPiece(unk_token="[UNK]", continuation_token="##")
            elif self.tokenization_method == "BPE":
                model = models.BPE(unk_token="[UNK]")
            else:
                raise ValueError("Invalid tokenization method. Choose 'WordPiece' or 'BPE'.")

            tokenizer = Tokenizer(model)
            tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
            trainer = trainers.WordPieceTrainer(
                vocab_size = vocab_size,
                special_tokens=self.special_tokens,
                initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            ) if self.tokenization_method == "WordPiece" else trainers.BpeTrainer(
                vocab_size = vocab_size,
                special_tokens=self.special_tokens,
                initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            )

            with open(self.training_file_path, "r", encoding="utf-8") as f:
                lines = [line.strip() for line in f]
                tokenizer.train_from_iterator(lines, trainer)
                
            self.tokenizer = tokenizer

            #self.save_vocab_to_file("vocab.txt")
    def save_tokenizer(self, output_file_path):
        self.tokenizer.save(str(Path(output_file_path) / "tokenizer.tk"))

    def save_vocab_to_file(self, output_file_path):
        vocab = self.tokenizer.get_vocab()
        sorted_vocab = sorted(vocab.items(), key=lambda item: item[1])

        with open(output_file_path, "w", encoding="utf-8") as f:
            for token, _ in sorted_vocab:
                f.write(f"{token}\n")

    def encode(self, text, max_length=None, padding='max_length', truncation=True):
        encoding = self.tokenizer.encode(text)
        if max_length is not None:
            encoding.pad(max_length, padding_id=self.tokenizer.token_to_id("[PAD]"), direction=padding)
            if truncation:
                encoding.truncate(max_length)
        return encoding


    def batch_encode(self, texts, max_length=None):
        encodings = [self.encode(text) for text in texts]
        if (max_length):
            for encoding in encodings:
                encoding.pad(max_length, pad_id = self.tokenizer.token_to_id("[PAD]"), direction="right")
                encoding.truncate(max_length)
        encoding_dict = {"input_ids":[], "attention_mask":[]}
        for encoding in encodings:
            encoding_dict["input_ids"].append(np.array(encoding.ids).reshape(1, -1))
            encoding_dict["attention_mask"].append(np.array(encoding.attention_mask).reshape(1, -1))
        encoding_dict["input_ids"] = np.concatenate(encoding_dict["input_ids"])
        encoding_dict["attention_mask"] = np.concatenate(encoding_dict["attention_mask"])
        return encoding_dict
    
    def _encode_with_padding_and_truncation(self, text, max_length):
        encoding = self.encode(text)
        if max_length is not None:
            encoding.pad(max_length, pad_id=self.tokenizer.token_to_id("[PAD]"), direction="right")
            encoding.truncate(max_length)
                
        return encoding

    def batch_encode_mp(self, texts, max_length=None, padding='max_length', truncation=True):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            encodings = list(executor.map(
                self._encode_with_padding_and_truncation,
                texts,
                [max_length] * len(texts),
            ))

        encoding_dict = {"input_ids": [], "attention_mask": []}
        for encoding in encodings:
            encoding_dict["input_ids"].append(np.array(encoding.ids).reshape(1, -1))
            encoding_dict["attention_mask"].append(np.array(encoding.attention_mask).reshape(1, -1))
        encoding_dict["input_ids"] = np.concatenate(encoding_dict["input_ids"])
        encoding_dict["attention_mask"] = np.concatenate(encoding_dict["attention_mask"])
        return encoding_dict

    def add_special_tokens(self, tokens):
        for token in tokens:
            if token not in self.special_tokens:
                self.special_tokens.append(token)
                self.tokenizer.add_special_tokens([token])

    def convert_tokens_to_ids(self, tokens):
        return self.tokenizer.token_to_id(tokens)

    def convert_ids_to_tokens(self, ids):
        return self.tokenizer.id_to_token(ids)

    def get_vocab_size(self):
        return self.tokenizer.get_vocab_size()
    
    def get_initial_embeddings(self, emb_dim, max_context_len):
        word_embeddings = nn.Embedding(self.get_vocab_size(), emb_dim)
        segment_embeddings = nn.Embedding(3, emb_dim)
        pos_embeddings = nn.Embedding(max_context_len, emb_dim)
        return word_embeddings, pos_embeddings, segment_embeddings
    
    def tokenize(
            self,
            text_embedder,
            position_embedder,
            seqment_embedder,
            texts = None,
            motion_embs = None,
            motion_masks = None,
            max_context_len = 128,
            device = "cpu",
            time_stats = None,
            base_key = ""
        ):
        #  CLS[0] SOT[1] Tokens[2 to C - (3 + m) + 1] MOT[C - (3 + m) + 2] Motions[C - (3 + m) + 3 to C - 1]
        if (time_stats is not None):
            tic(time_stats, base_key + "/Text Embs/Pre")
        #seq_emb = seqment_embedder(torch.tensor([0], device = device))

        sample_len = len(texts)
        reserved_motion_len = motion_masks.shape[1] + 1 if motion_embs is not None else 0
        max_token_len = max_context_len - (2 + reserved_motion_len)
        if (time_stats is not None):
            tic(time_stats, base_key + "/Text Embs/Embed")
        # Tokenize texts and pad to max_token_len
        #texts = len(texts) * ["[PAD]"]
        text_encodings = self.batch_encode_mp(texts, max_length=max_token_len)
        text_ids = torch.tensor(text_encodings["input_ids"], device=device)
        text_mask = torch.tensor(text_encodings["attention_mask"], device=device)

        # Pad text_ids and text_mask tensors to ensure they always have the shape (batch_size, max_token_len)
        text_ids = torch.cat([text_ids, torch.zeros(text_ids.shape[0], max_token_len - text_ids.shape[1], dtype=torch.long, device=device)], dim=1)
        text_mask = torch.cat([text_mask, torch.zeros(text_mask.shape[0], max_token_len - text_mask.shape[1], dtype=torch.long, device=device)], dim=1)

        # Embed tokens and concatenate
        text_embs = text_embedder(text_ids)

        if (time_stats is not None):
            tic(time_stats, base_key + "/Text Embs/SPT")

        text_start_token = text_embedder(torch.tensor([self.convert_tokens_to_ids("[SOT]")], device=device)).repeat(sample_len, 1, 1)
        text_embs = torch.cat([text_start_token, text_embs], dim=1)
        if (self.use_seg_embs):
            seq_emb = seqment_embedder(torch.tensor([0], device = device))
            text_embs += seq_emb
        if (time_stats is not None):
            tic(time_stats, base_key + "/Text Embs/Mask")
        # Create mask
        mask = torch.zeros((sample_len, max_context_len), dtype=torch.bool, device=device)
        mask[:, :2] = True
        mask[:, 2: max_token_len + 2] = text_mask

        if (time_stats is not None):
            tic(time_stats, base_key + "/Misc")
        if (motion_embs is not None):
            sample_len = max(len(motion_embs), sample_len)
        cls_emb = text_embedder(torch.tensor([self.convert_tokens_to_ids("[CLS]")], device = device))[None, ...].repeat(sample_len, 1, 1)

        final_embs = torch.cat([cls_emb, text_embs], dim=1)
        
        if (time_stats is not None):
            tic(time_stats, base_key + "/Motion Embs")
        if (motion_embs is not None):

            start_token = text_embedder(torch.tensor([self.convert_tokens_to_ids("[MOT]")], device = device)).repeat(sample_len, 1, 1)
            mask[:, final_embs.shape[1]] = True
            mask[:, final_embs.shape[1]+1:] = motion_masks
            motion_embs = torch.cat([start_token, motion_embs], dim=1)# + seq_emb
            if (self.use_seg_embs):
                seq_emb = seqment_embedder(torch.tensor([1], device = device))
                motion_embs += seq_emb
            final_embs = torch.cat([final_embs, motion_embs], dim = 1)
            
            
        if (time_stats is not None):
            tic(time_stats, base_key + "/Pos Embs")
        final_embs = final_embs + position_embedder(torch.arange(0, max_context_len, device = device))[None, ...].repeat(sample_len, 1, 1)
        #print(torch.mean(final_embs[0], dim=0))
        return final_embs, mask, text_ids, text_mask

def load_pickle(filepath):
    with open(filepath, "rb") as f:
        return pickle.load(f)

class MotionTextEvalBERT(nn.Module):
        def __init__(self, primary_evaluator_model_config, chunk_encoder_config, tokenizer_and_embedders_config, tokenizer_path="", load_trained_regressors_path = ""):
            super().__init__()
            self.num_heads = primary_evaluator_model_config["num_heads"]
            self.num_layers = primary_evaluator_model_config["num_layers"]
            self.enc_dim = primary_evaluator_model_config["enc_dim"]
            self.max_context_len = tokenizer_and_embedders_config["max_context_length"]
            self.use_seg_embs = tokenizer_and_embedders_config["use_seg_embs"]
            self.tokenizer_train_method = tokenizer_and_embedders_config["train_method"]
            self.vocab_size = tokenizer_and_embedders_config["vocab_size"]
            self.tokenizer_train_path = tokenizer_and_embedders_config["tokenizer_train_path"]
            self.group_norm = primary_evaluator_model_config["group_norm"]
            self.load_trained_regressors_path = load_trained_regressors_path
            if (self.load_trained_regressors_path):
                self.naturalness_regressor = load_pickle(Path(self.load_trained_regressors_path) / "SVR_PROB_CLS_MOTIONTEXT_Naturalness.obj")
                self.faithfulness_regressor = load_pickle(Path(self.load_trained_regressors_path) / "SVR_PROB_CLS_MOTIONTEXT_Faithfulness.obj")

            self.frame_encoder = nn.Sequential(
                nn.Linear(chunk_encoder_config["input_dim"], chunk_encoder_config["frame_enc_dim"]),
                nn.GroupNorm(chunk_encoder_config["frame_enc_groups"], chunk_encoder_config["frame_enc_dim"]),
                nn.SELU(),
                nn.Dropout(float(chunk_encoder_config["dropout"])),
                nn.Linear(chunk_encoder_config["frame_enc_dim"], chunk_encoder_config["frame_enc_dim"]),
                nn.GroupNorm(chunk_encoder_config["frame_enc_groups"], chunk_encoder_config["frame_enc_dim"]),
                nn.SELU(),
                nn.Dropout(float(chunk_encoder_config["dropout"])),
            )
            
            self.chunk_encoder = nn.Sequential(
                nn.Linear(chunk_encoder_config["frame_enc_dim"] * (chunk_encoder_config["chunk_size"] + chunk_encoder_config["chunk_overlap"]), chunk_encoder_config["frame_enc_dim"] * (chunk_encoder_config["chunk_size"] + chunk_encoder_config["chunk_overlap"])),
                nn.GroupNorm((chunk_encoder_config["chunk_size"] + chunk_encoder_config["chunk_overlap"]), chunk_encoder_config["frame_enc_dim"] * (chunk_encoder_config["chunk_size"] + chunk_encoder_config["chunk_overlap"])),
                nn.SELU(),
                nn.Dropout(float(chunk_encoder_config["dropout"])),
                nn.Linear(chunk_encoder_config["frame_enc_dim"] * (chunk_encoder_config["chunk_size"] + chunk_encoder_config["chunk_overlap"]), chunk_encoder_config["frame_enc_red_dim"] * (chunk_encoder_config["chunk_size"] + chunk_encoder_config["chunk_overlap"])),
                nn.GroupNorm(chunk_encoder_config["frame_enc_groups"], chunk_encoder_config["frame_enc_red_dim"] * (chunk_encoder_config["chunk_size"] + chunk_encoder_config["chunk_overlap"])),
                nn.SELU(),
                nn.Dropout(float(chunk_encoder_config["dropout"])),
                nn.Linear(chunk_encoder_config["frame_enc_red_dim"] * (chunk_encoder_config["chunk_size"] + chunk_encoder_config["chunk_overlap"]), chunk_encoder_config["enc_dim"]),
                nn.GroupNorm(chunk_encoder_config["frame_enc_groups"], chunk_encoder_config["enc_dim"]),
                nn.SELU(),
                nn.Dropout(float(chunk_encoder_config["dropout"])),
                nn.Linear(chunk_encoder_config["enc_dim"], primary_evaluator_model_config["enc_dim"]),
                nn.GroupNorm(chunk_encoder_config["frame_enc_groups"], primary_evaluator_model_config["enc_dim"]),
                nn.SELU(),
                nn.Dropout(float(chunk_encoder_config["dropout"])),
            )

            self.tokenizer = TMEvalTokenizer(self.tokenizer_train_method, self.tokenizer_train_path, vocab_size=self.vocab_size, saved_tokenizer_path=tokenizer_path, use_seg_embs = self.use_seg_embs)
            self.word_embs, self.pos_embs, self.seq_embs = self.tokenizer.get_initial_embeddings(primary_evaluator_model_config["enc_dim"], tokenizer_and_embedders_config["max_context_length"]) 
            encoder_layer = nn.TransformerEncoderLayer(d_model=self.enc_dim, nhead=self.num_heads, dropout=float(primary_evaluator_model_config["dropout"]), dim_feedforward=primary_evaluator_model_config["dim_ff"], batch_first = True)
            self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)
            
            self.token_pred_head = nn.Sequential(
                nn.Linear(self.enc_dim, primary_evaluator_model_config["token_pred_dim_ff"]),
                nn.GroupNorm(chunk_encoder_config["frame_enc_groups"], primary_evaluator_model_config["token_pred_dim_ff"]),
                nn.SELU(),
                nn.Dropout(float(chunk_encoder_config["dropout"])),
                nn.Linear(primary_evaluator_model_config["token_pred_dim_ff"], primary_evaluator_model_config["token_pred_dim_ff"]),
                nn.GroupNorm(chunk_encoder_config["frame_enc_groups"], primary_evaluator_model_config["token_pred_dim_ff"]),
                nn.SELU(),
                nn.Dropout(float(chunk_encoder_config["dropout"])),
                nn.Linear(primary_evaluator_model_config["token_pred_dim_ff"], self.tokenizer.get_vocab_size()),
                nn.LogSoftmax(dim=-1)
                )
            self.ent_pred_head = nn.Sequential(
                nn.Linear(self.enc_dim, primary_evaluator_model_config["token_pred_dim_ff"]),
                nn.GroupNorm(chunk_encoder_config["frame_enc_groups"], primary_evaluator_model_config["token_pred_dim_ff"]),
                nn.SELU(),
                nn.Dropout(float(chunk_encoder_config["dropout"])),
                nn.Linear(primary_evaluator_model_config["token_pred_dim_ff"], primary_evaluator_model_config["token_pred_dim_ff"]),
                nn.GroupNorm(chunk_encoder_config["frame_enc_groups"], primary_evaluator_model_config["token_pred_dim_ff"]),
                nn.SELU(),
                nn.Dropout(float(chunk_encoder_config["dropout"])),
                nn.Linear(primary_evaluator_model_config["token_pred_dim_ff"], 1)
                )
            
            self.use_sentence_transformer = primary_evaluator_model_config["use_sentence_transformer"]
            self.st_distance = primary_evaluator_model_config["st_distance"]
            self.st_model = primary_evaluator_model_config["st_model"]
            self.embedding_cache = {}

        def encode_motion_chunks(self, motion_chunks):
            orig_shape = motion_chunks.shape
            if (self.group_norm):
                embs = motion_chunks.view(-1, orig_shape[-1])
                embs = self.frame_encoder(embs)
                embs = embs.view(-1, embs.shape[-1] * orig_shape[-2])
                embs = self.chunk_encoder(embs)
                embs = embs.view(orig_shape[0], orig_shape[1], -1)
            else:
                embs = motion_chunks.view(orig_shape[0], orig_shape[1], -1)
                embs = self.frame_encoder(embs)
                embs = self.chunk_encoder(embs)
            return embs
        
        def rate_alignment_batch(self, texts, motion_chunks, motion_masks, device):
            motion_embs = self.encode_motion_chunks(motion_chunks)
            token_embs, attn_masks, text_ids, text_mask = self.tokenizer.tokenize(
                self.word_embs, self.pos_embs, self.seq_embs, 
                texts, 
                motion_embs, motion_masks,
                max_context_len = self.max_context_len,
                device=device,
            )
            res_embs = self.transformer_encoder(token_embs, src_key_padding_mask=~attn_masks)
            cls_embs = res_embs[:, 0, ...]
            text_acum_embs = torch.max((res_embs * attn_masks[..., None])[:, 2:self.max_context_len - motion_embs.shape[1] - 1, :], dim=1).values
            motion_acum_embs = torch.max((res_embs * attn_masks[..., None])[:, self.max_context_len - motion_embs.shape[1]:, :], dim=1).values
            alingment_preds = self.ent_pred_head(cls_embs)

            regression_features = torch.concat([alingment_preds, cls_embs, text_acum_embs, motion_acum_embs], dim=1)
            if (self.load_trained_regressors_path):
                naturalness_rating = self.naturalness_regressor.predict(regression_features.detach().cpu().numpy()) / 4
                faithfulness_rating = self.faithfulness_regressor.predict(regression_features.detach().cpu().numpy()) / 4
            return nn.functional.sigmoid(alingment_preds.view(-1)), faithfulness_rating, naturalness_rating


        
        def motion_text_mask_prediction(
                self, 
                texts, 
                motion_embs, motion_masks,
                num_mask,
                device,
                time_stats = None
                ):
            if (time_stats is not None):
                tic(time_stats, "NLL/Mask Pred/Tokenize")
            token_embs, attn_masks, text_ids, text_mask = self.tokenizer.tokenize(
                self.word_embs, self.pos_embs, self.seq_embs, 
                texts, 
                motion_embs, motion_masks,
                max_context_len = self.max_context_len,
                device=device,
                time_stats = time_stats,
                base_key = "NLL/Mask Pred/Tokenize"
            )
            
            if (time_stats is not None):
                tic(time_stats, "NLL/Mask Pred/ATTN")
            mask_emb = self.word_embs(torch.tensor([], device = device))

            if (time_stats is not None):
             tic(time_stats, "NLL/Mask Pred/Mask Proc")
            masked_inds = []
            masked_embs = []
            masked_ids = []
            max_m = torch.sum(text_mask, dim=1) - 1
            for i in range(len(token_embs)):
                i_masked_inds = []
                for k in range(num_mask):
                    
                    mask_m = random.randint(0, max_m[i])
                    i_masked_inds.append(mask_m)
                    masked_ids.append(text_ids[i][mask_m])
                    masked_embs.append(token_embs[i, k + 2, ...].detach()[None, ...])
                    token_embs[i, k + 2, ...] = mask_emb
                masked_inds.append(i_masked_inds)
            masked_embs = torch.cat(masked_embs, dim=0)
            masked_ids = torch.tensor(masked_ids, device=device, dtype=torch.long)
            if (time_stats is not None):
                tic(time_stats, "NLL/Mask Pred/Transformer Forward")
            res_embs = self.transformer_encoder(token_embs, src_key_padding_mask=~attn_masks)
            if (time_stats is not None):
                tic(time_stats, "NLL/Mask Pred/Res Aggregate")
            res_masked_embs = []
            for i in range(len(masked_inds)):
                for j in range(len(masked_inds[i])):
                    res_masked_embs.append(res_embs[i, masked_inds[i][j] + 2][None, ...])
            res_masked_embs = torch.cat(res_masked_embs, dim=0)
            if (time_stats is not None):
                tic(time_stats, "NLL/Mask Pred/Pred Head")
            res_masked_preds = self.token_pred_head(res_masked_embs)

            nll_loss_func = nn.NLLLoss()

            nll_loss = nll_loss_func(res_masked_preds, masked_ids)
            return nll_loss
        
        def get_prob_and_cls_prediction(
                self, 
                texts, 
                motion_embs, motion_masks,
                device
                ):
            
            valid_token_embs, valid_attn_masks, _, _ = self.tokenizer.tokenize(
                self.word_embs, self.pos_embs, self.seq_embs, 
                texts, 
                motion_embs, motion_masks,
                max_context_len = self.max_context_len,
                device=device
            )
            
            valid_res_embs = self.transformer_encoder(valid_token_embs, src_key_padding_mask=~valid_attn_masks)
            valid_cls_embs = valid_res_embs[:, 0, ...]
            valid_sot_embs = valid_res_embs[:, 1, ...]
            valid_mot_embs = valid_res_embs[:, self.max_context_len - motion_embs.shape[1] - 1, ...]
            valid_text_embs = torch.max((valid_res_embs * valid_attn_masks[..., None])[:, 2:self.max_context_len - motion_embs.shape[1] - 1, :], dim=1).values
            valid_motion_embs = torch.max((valid_res_embs * valid_attn_masks[..., None])[:, self.max_context_len - motion_embs.shape[1]:, :], dim=1).values
            valid_text_sembs = torch.sum((valid_res_embs * valid_attn_masks[..., None])[:, 2:self.max_context_len - motion_embs.shape[1] - 1, :], dim=1)
            valid_motion_sembs = torch.sum((valid_res_embs * valid_attn_masks[..., None])[:, self.max_context_len - motion_embs.shape[1]:, :], dim=1)

            stage_pred_embs = {}
            for i in range(0, len(self.ent_pred_head) - 3, 3):
                stage_pred_embs[f"PREDS{i // 3 + 1}"] = stage_pred_embs[f"PREDS{i // 3}"] if i > 0 else valid_cls_embs
                for j in range(i, i+3):
                    stage_pred_embs[f"PREDS{i // 3 + 1}"] = self.ent_pred_head[j](stage_pred_embs[f"PREDS{i // 3 + 1}"])
            
            valid_ent_preds = self.ent_pred_head[-1](stage_pred_embs[f"PREDS{i // 3 + 1}"]).view(-1)

            return valid_ent_preds, valid_cls_embs, valid_sot_embs, valid_mot_embs, valid_text_embs, valid_motion_embs, valid_text_sembs, valid_motion_sembs, stage_pred_embs, valid_res_embs
        
        def get_sentence_embeddings(self, texts, device):
            embeddings = {}
            unseen_texts = []
            unseen_idx = []

            for i, text in enumerate(texts):
                if text in self.embedding_cache:
                    embeddings[i] = self.embedding_cache[text]
                else:
                    unseen_texts.append(text)
                    unseen_idx.append(i)

            if unseen_texts:
                sentence_transformer = SentenceTransformer(self.st_model).to(device)
                unseen_embeddings = sentence_transformer.encode(unseen_texts)
                for text, emb in zip(unseen_texts, unseen_embeddings):
                    self.embedding_cache[text] = emb

                for idx, emb in zip(unseen_idx, unseen_embeddings):
                    embeddings[idx] = emb
            
            embeddings = [embeddings[i] for i in range(len(texts))]

            return torch.tensor(np.stack(embeddings, axis=0), device=device)

        def ent_prediction(
                self, 
                texts,
                rand_texts, 
                motion_embs, motion_masks,
                device,
                time_stats = None
                ):
            
            bce_loss_func = nn.BCEWithLogitsLoss()
            if (time_stats is not None):
                tic(time_stats, "Entail/Run Entail/Valid Tokenize")
            valid_token_embs, valid_attn_masks, _, _ = self.tokenizer.tokenize(
                self.word_embs, self.pos_embs, self.seq_embs, 
                texts, 
                motion_embs, motion_masks,
                max_context_len = self.max_context_len,
                device=device,
                time_stats = time_stats,
                base_key = "Entail/Run Entail/Valid Tokenize"
            )
            if (time_stats is not None):
                tic(time_stats, "Entail/Run Entail/Valid Transformer Forward")
            valid_res_embs = self.transformer_encoder(valid_token_embs, src_key_padding_mask=~valid_attn_masks)
            if (time_stats is not None):
                tic(time_stats, "Entail/Run Entail/Valid Res Aggregate")
            valid_cls_embs = valid_res_embs[:, 0, ...]
            if (time_stats is not None):
                tic(time_stats, "Entail/Run Entail/Valid Pres Head")
            valid_ent_preds = self.ent_pred_head(valid_cls_embs).view(-1)
            if (time_stats is not None):
                tic(time_stats, "Entail/Run Entail/Valid Loss")
            valid_ent_labels = torch.ones(len(valid_ent_preds), device = device)
            valid_ent_loss = bce_loss_func(valid_ent_preds, valid_ent_labels)
            if (time_stats is not None):
                tic(time_stats, "Entail/Run Entail/Rand Tokenize")
            rand_token_embs, rand_attn_masks, _, _ = self.tokenizer.tokenize(
                self.word_embs, self.pos_embs, self.seq_embs, 
                rand_texts, 
                motion_embs, motion_masks,
                max_context_len = self.max_context_len,
                device=device,
                time_stats = time_stats,
                base_key = "Entail/Run Entail/Rand Tokenize"
            )
            loss_func = nn.BCEWithLogitsLoss(reduction='none')
            if (time_stats is not None):
                tic(time_stats, "Entail/Run Entail/Rand Transformer Forward")
            rand_res_embs = self.transformer_encoder(rand_token_embs, src_key_padding_mask=~rand_attn_masks)
            if (time_stats is not None):
                tic(time_stats, "Entail/Run Entail/Rand Res Aggregate")
            rand_cls_embs = rand_res_embs[:, 0, ...]
            if (time_stats is not None):
                tic(time_stats, "Entail/Run Entail/Rand Pres Head")
            rand_ent_preds = self.ent_pred_head(rand_cls_embs).view(-1)

            rand_ent_labels = torch.zeros(len(rand_ent_preds), device=device)
            scaled_weights = torch.ones(len(rand_ent_labels), device=device)
            mean_scaled_weight = 1
            if (time_stats is not None):
                tic(time_stats, "Entail/Run Entail/Rand Weight Proc")
            if self.use_sentence_transformer:
                rand_sent_embeddings = self.get_sentence_embeddings(rand_texts, device)
                valid_sent_embeddings = self.get_sentence_embeddings(texts, device)

                # Compute inverse similarity for each sample in the batch
                if self.st_distance == 'euclidean':
                    distances = torch.norm(valid_sent_embeddings - rand_sent_embeddings, dim=1)
                else:  # cosine similarity
                    valid_text_embs_norm = nn.functional.normalize(valid_sent_embeddings, p=2, dim=1)
                    embeddings_norm = nn.functional.normalize(rand_sent_embeddings, p=2, dim=1)
                    cos_similarities = torch.sum(valid_text_embs_norm * embeddings_norm, dim=1)
                    distances = 1 - cos_similarities

                scaled_weights = distances
                mean_scaled_weight = torch.mean(scaled_weights) + 1e-8
            if (time_stats is not None):
                tic(time_stats, "Entail/Run Entail/Rand Loss")
            if self.use_sentence_transformer:
                rand_ent_loss = torch.mean(loss_func(rand_ent_preds, rand_ent_labels) * scaled_weights) / mean_scaled_weight
            else:
                rand_ent_loss = bce_loss_func(rand_ent_preds, rand_ent_labels)

            return valid_ent_loss, rand_ent_loss
        
