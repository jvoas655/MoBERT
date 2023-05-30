import argparse
import os
import random
import yaml
from pathlib import Path
import torch
import numpy
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from dataset.primary_evaluator_dataset import *
from torch.utils.data import DataLoader
from models.motion_text_eval_bert import *
from torch.optim import lr_scheduler
from torch.nn.utils import clip_grad_norm_
import scipy.stats as scistat
from sklearn.model_selection import KFold
from sklearn.linear_model import *
from sklearn.svm import SVR
from utils.stat_tracking import *

import itertools

def generate_fit_feature_combinations(fit_features):
    feature_keys = list(fit_features.keys())
    combinations = []

    for i in range(1, len(feature_keys) + 1):
        for combo in itertools.combinations(feature_keys, i):
            combinations.append(combo)

    return combinations

#os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def save_pickle(obj, filepath):
    with open(filepath, "wb") as f:
        pickle.dump(obj, f)

def load_pickle(filepath):
    with open(filepath, "rb") as f:
        return pickle.load(f)

        
def parse_arguments():
    parser = argparse.ArgumentParser(description = "Process arguments for initial training of motion chunk encoder")
    parser.add_argument("--device", "-d", default = 0, type = str, help = "Device to utilize (-1 = CPU) (> - 1 = GPU)")
    parser.add_argument("--config", "-c", default = Path("configs/base_config.yml"))
    parser.add_argument("--resume", "-r", default="", help="Checkpoint to resume training from. ")
    parser.add_argument("--reset_scheduler", action="store_true")
    args = parser.parse_args()
    with open(args.config, "r") as conf:
        config = yaml.safe_load(conf)
    return args, config["primary_evaluator"]

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    

if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = True
    args, config = parse_arguments()
    seed = config["seed"]
    set_seed(seed)
    os.makedirs(Path(config["log_path"]) / config["exp_name"], exist_ok = True)
    os.makedirs(Path(config["checkpoint_path"]) / config["exp_name"], exist_ok = True)
    writer = SummaryWriter(Path(config["log_path"]) / config["exp_name"], comment=config["exp_name"])
    chunk_encoder_config = config["chunk_encoder"]
    tokenizer_and_embedders_config = config["tokenizer_and_embedders"]
    primary_evaluator_model_config = config["primary_evaluator_model"]
    training_config = config["training_params"]

    device = "cpu"
    if (torch.cuda.is_available() and int(args.device) >= 0):
        device = f"cuda:{args.device}"
    model = MotionTextEvalBERT(primary_evaluator_model_config, chunk_encoder_config, tokenizer_and_embedders_config)
    model.tokenizer.save_tokenizer(Path(config["checkpoint_path"]) / config["exp_name"])
    #model = torch.compile(model, mode="max-autotune")
    print(model)
    model = model.to(device = device)
    dval = TMRefDataset(config["cache_key"], "val", Path(config["val_path"]), training_config["max_val_samples"], chunk_encoder_config["chunk_size"], chunk_encoder_config["chunk_overlap"], 1 + (200 // (chunk_encoder_config["chunk_size"])), training_config["num_threads"], augment=False)
    val_data = DataLoader(dval, batch_size=training_config["batch_size"], num_workers=training_config["num_threads"], shuffle=True, drop_last=True)
    dtrain = TMRefDataset(config["cache_key"], "train", Path(config["train_path"]), training_config["max_train_samples"], chunk_encoder_config["chunk_size"], chunk_encoder_config["chunk_overlap"], 1 + (200 // (chunk_encoder_config["chunk_size"])), training_config["num_threads"])
    train_data = DataLoader(dtrain, batch_size=training_config["batch_size"], num_workers=training_config["num_threads"], shuffle=True, drop_last=True)
    dtest_judge = JudgementDataset(Path(config["cache_key"]) / "cache_eval", Path("../MotionDataset/evaluation_data/"), Path(config["val_path"]), chunk_encoder_config["chunk_size"], chunk_encoder_config["chunk_overlap"], 1 + (200 // (chunk_encoder_config["chunk_size"] + chunk_encoder_config["chunk_overlap"])), device=device)
    dtest_judge_data = DataLoader(dtest_judge, batch_size=training_config["batch_size"], num_workers=8, shuffle=False, drop_last=False)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(training_config["learning_rate"]))
    if (float(training_config["ae_learning_rate"]) > 0.0):
        ae_optimizer = torch.optim.AdamW(model.parameters(), lr=float(training_config["ae_learning_rate"]))
    num_steps = int(len(train_data) * float(training_config["num_epochs"]))
    scheduler_ca = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 1000, 2)

    regression_types = {
            "SVRRBF": SVR(kernel="rbf", epsilon=0.3, tol=1e-8, C=3.68, max_iter=int(1e4)), 
        }
    metric_types = ["Naturalness", "Faithfulness"]
    
    epoch = 1
    last_iter = 0
    iters = 1
    text_nll_loss_acum = 0
    valid_ent_loss_acum = 0
    rand_ent_loss_acum = 0
    loss_acum = 0
    best_corr_values = {"Naturalness": -1 * float("inf"), "Faithfulness": -1 * float("inf")}
    best_corr_keys = {"Naturalness": "", "Faithfulness": ""}
    pbar = tqdm.tqdm(total=num_steps)

    if (args.resume != ""):
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        
        iters = checkpoint["iters"]
        last_iter = iters - 1
        epoch = checkpoint["epoch"]
        if "best_faithfulness_corr_value" in checkpoint:
            best_corr_values["Faithfulness"] = checkpoint["best_faithfulness_corr_value"]
        if "best_naturalness_corr_value" in checkpoint:
            best_corr_values["Naturalness"] = checkpoint["best_naturalness_corr_value"]
        if (not args.reset_scheduler):
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            scheduler_ca.load_state_dict(checkpoint["scheduler_ca"])

    if (primary_evaluator_model_config["st_emb_cache"] != "" and os.path.exists(Path(primary_evaluator_model_config["st_emb_cache"]))):
        model.embedding_cache = load_pickle(Path(primary_evaluator_model_config["st_emb_cache"]))

    
    pbar.update(iters - 1)
    val_rand_state = random.getstate()
    time_stats = {}
    
    while (iters <= num_steps):
        model.train()
        
        for batch_i, batch_data in enumerate(train_data):
            batch_motion_chunks = batch_data["motion_chunks"].to(device = device, dtype=torch.float32)
            batch_motion_masks = batch_data["motion_masks"].to(device = device, dtype=torch.float32)
            batch_motion_texts = batch_data["texts"]
            batch_motion_rand_texts = batch_data["random_texts"]
            
            time_stats = {}
            optimizer.zero_grad()
            tic(time_stats, "Entail/Encode Motion")
            batch_motion_chunk_embs = model.encode_motion_chunks(batch_motion_chunks)
            tic(time_stats, "Entail/Run Entail")
            valid_ent_loss, rand_ent_loss = model.ent_prediction(
                batch_motion_texts,
                batch_motion_rand_texts,
                batch_motion_chunk_embs, batch_motion_masks,
                device=device,
                time_stats = time_stats
                )
            loss = float(training_config["entail_weight"]) * torch.sqrt(torch.pow(valid_ent_loss, 2) + torch.pow(rand_ent_loss, 2))
            tic(time_stats, "Entail/Opt Step")
            loss.backward()
            clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            valid_ent_loss_acum += valid_ent_loss.detach().item()
            rand_ent_loss_acum += rand_ent_loss.detach().item()
            loss_acum += loss.detach().item()

            if (float(training_config["nll_weight"]) > 0.0):
                optimizer.zero_grad()
                tic(time_stats, "NLL/Encode Motion")
                batch_motion_chunk_embs = model.encode_motion_chunks(batch_motion_chunks)
                tic(time_stats, "NLL/Mask Pred")
                text_nll_loss = model.motion_text_mask_prediction(
                    batch_motion_texts,
                    batch_motion_chunk_embs, batch_motion_masks,
                    1,
                    device=device,
                    time_stats = time_stats
                    )
                loss = float(training_config["nll_weight"]) * text_nll_loss
                tic(time_stats, "NLL/Opt Step")
                loss.backward()
                clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                text_nll_loss_acum += text_nll_loss.detach().item()
                loss_acum += loss.detach().item()
            toc(time_stats)
            scheduler_ca.step()
            
            pbar.update(1)
            pbar.set_description(
                f"Training   {epoch} Loss {'{:.3e}'.format(loss_acum / (iters - last_iter)) }"
            )
            if (iters % training_config["val_iters"] == 0):
                writer.add_scalar("Loss/train/sum", loss_acum / (iters - last_iter), iters)

                if (float(training_config["nll_weight"]) > 0.0):
                    writer.add_scalar("Loss/train/NLL", text_nll_loss_acum / (iters - last_iter), iters)
                writer.add_scalar("Loss/train/VENT", valid_ent_loss_acum / (iters - last_iter), iters)
                writer.add_scalar("Loss/train/RENT", rand_ent_loss_acum / (iters - last_iter), iters)
                text_nll_loss_acum = 0
                valid_ent_loss_acum = 0
                rand_ent_loss_acum = 0
                loss_acum = 0
                train_rand_state = random.getstate()
                random.setstate(val_rand_state)
                model.eval()
                with torch.no_grad():
                    
                    for batch_i, batch_data in enumerate(val_data):
                        batch_motion_chunks = batch_data["motion_chunks"].to(device = device, dtype=torch.float32)
                        batch_motion_masks = batch_data["motion_masks"].to(device = device, dtype=torch.float32)
                        batch_motion_texts = batch_data["texts"]
                        batch_motion_rand_texts = batch_data["random_texts"]
                        
                        time_stats = {}
                        optimizer.zero_grad()
                        batch_motion_chunk_embs = model.encode_motion_chunks(batch_motion_chunks)
                        valid_ent_loss, rand_ent_loss = model.ent_prediction(
                            batch_motion_texts,
                            batch_motion_rand_texts,
                            batch_motion_chunk_embs, batch_motion_masks,
                            device=device
                            )
                        loss = float(training_config["entail_weight"]) * torch.sqrt(torch.pow(valid_ent_loss, 2) + torch.pow(rand_ent_loss, 2))
                        valid_ent_loss_acum += valid_ent_loss.detach().item()
                        rand_ent_loss_acum += rand_ent_loss.detach().item()
                        loss_acum += loss.detach().item()
                    writer.add_scalar("Loss/val/sum", loss_acum / (batch_i + 1), iters)
                    writer.add_scalar("Loss/val/VENT", valid_ent_loss_acum / (batch_i + 1), iters)
                    writer.add_scalar("Loss/val/RENT", rand_ent_loss_acum / (batch_i + 1), iters)
                    
                    writer.add_scalar("LR", scheduler_ca.get_last_lr()[0], iters)
                    all_probs = []
                    all_cls = []
                    all_sot = []
                    all_mot = []
                    all_text = []
                    all_motion = []
                    all_stext = []
                    all_smotion = []
                    all_pred_embs = {}
                    all_faithfullness = []
                    all_naturalness = []
                    for batch_i, batch_data in enumerate(dtest_judge_data):
                        batch_motion_chunks = batch_data["motion_chunks"].to(device = device, dtype=torch.float32)
                        batch_motion_masks = batch_data["motion_masks"].to(device = device, dtype=torch.float32)
                        batch_motion_texts = batch_data["texts"]
                        batch_motion_faithfulness = batch_data["faithfulness"]
                        batch_motion_naturalness = batch_data["naturalness"]
                        
                        batch_motion_chunk_embs = model.encode_motion_chunks(batch_motion_chunks)
                        probs, cls_embs, sot_embs, mot_embs, text_embs, motion_embs, text_sembs, motion_sembs, pred_embs, res_embs = model.get_prob_and_cls_prediction(
                            batch_motion_texts,
                            batch_motion_chunk_embs, batch_motion_masks,
                            device=device
                            )
                        all_probs.append(nn.functional.sigmoid(probs).detach().cpu().numpy())
                        all_cls.append(cls_embs.detach().cpu().numpy())
                        all_sot.append(sot_embs.detach().cpu().numpy())
                        all_mot.append(mot_embs.detach().cpu().numpy())
                        all_text.append(text_embs.detach().cpu().numpy())
                        all_motion.append(motion_embs.detach().cpu().numpy())
                        all_stext.append(text_sembs.detach().cpu().numpy())
                        all_smotion.append(motion_sembs.detach().cpu().numpy())
                        for key in pred_embs:
                            if (key not in all_pred_embs):
                                all_pred_embs[key] = []
                            all_pred_embs[key].append(pred_embs[key].detach().cpu().numpy())
                        all_faithfullness.append(batch_motion_faithfulness.detach().cpu().numpy())
                        all_naturalness.append(batch_motion_naturalness.detach().cpu().numpy())
                    all_probs = np.concatenate(all_probs, axis = 0).reshape(-1, 1)
                    all_cls = np.concatenate(all_cls, axis = 0)
                    all_sot = np.concatenate(all_sot, axis = 0)
                    all_mot = np.concatenate(all_mot, axis = 0)
                    all_text = np.concatenate(all_text, axis = 0)
                    all_motion = np.concatenate(all_motion, axis = 0)
                    all_stext = np.concatenate(all_stext, axis = 0)
                    all_smotion = np.concatenate(all_smotion, axis = 0)
                    for key in all_pred_embs:
                        all_pred_embs[key] = np.concatenate(all_pred_embs[key], axis = 0)
                    motsot_dot_product = all_mot * all_sot
                    mot_norm = np.linalg.norm(all_mot, axis=1).reshape(-1, 1)
                    sot_norm = np.linalg.norm(all_sot, axis=1).reshape(-1, 1)
                    all_motsot_sim = motsot_dot_product / (mot_norm * sot_norm)
                    all_motsot_cs = np.sum(all_motsot_sim, axis=1).reshape(-1, 1)
                    motiontext_dot_product = all_motion * all_text
                    motion_norm = np.linalg.norm(all_motion, axis=1).reshape(-1, 1)
                    text_norm = np.linalg.norm(all_text, axis=1).reshape(-1, 1)
                    all_motiontext_sim = motiontext_dot_product / (motion_norm * text_norm)
                    all_motiontext_cs = np.sum(all_motiontext_sim, axis=1).reshape(-1, 1)

                    smotiontext_dot_product = all_smotion * all_stext
                    smotion_norm = np.linalg.norm(all_smotion, axis=1).reshape(-1, 1)
                    stext_norm = np.linalg.norm(all_stext, axis=1).reshape(-1, 1)
                    all_smotiontext_sim = smotiontext_dot_product / (smotion_norm * stext_norm)
                    all_smotiontext_cs = np.sum(all_smotiontext_sim, axis=1).reshape(-1, 1)
                    all_faithfullness = np.concatenate(all_faithfullness, axis = 0)
                    all_naturalness = np.concatenate(all_naturalness, axis = 0)
                fit_features = {"CLS": all_cls, "MOTIONTEXT": np.concatenate([all_text, all_motion], axis=1), "PROB": all_probs}
                for key in all_pred_embs:
                    fit_features[key] = all_pred_embs[key]

                metric_data = {"Naturalness": all_naturalness, "Faithfulness": all_faithfullness}

                kf_test_inds = []
                all_unordered_res = {}
                res_dict = {}
                save_bools = {"Naturalness": False, "Faithfulness": False}
                best_corr = -1 * float("inf")
                num_fold = 4
                fit_feature_combinations = generate_fit_feature_combinations(fit_features)
                vbar = tqdm.tqdm(desc=f"Validation {best_corr}", total= num_fold * len(fit_feature_combinations) * len(regression_types))

                for corr_metric in metric_types:
                    if (corr_metric == "Naturalness"):
                        continue
                    res = scistat.pearsonr(all_probs.flatten(), metric_data[corr_metric])
                    p, corr = res.pvalue, res.statistic
                    res_dict[f"Perf/val/prob_{corr_metric}/corr"] = corr
                    res_dict[f"Perf/val/prob_{corr_metric}/pval"] = p
                    if (corr > best_corr_values[corr_metric]):
                        best_corr_keys[corr_metric] = f"prob"
                        best_corr_values[corr_metric] = corr
                        save_bools[corr_metric] = True
                    if (corr > best_corr):
                        best_corr = corr
                        vbar.set_description(f"Validation {best_corr}")
    
                # CLS
                kf = KFold(n_splits = num_fold, shuffle=True, random_state = 42)
                fold_inds = kf.split(all_cls, all_faithfullness)
                
                
                for i, (train_split, test_split) in enumerate(fold_inds):
                    kf_test_inds.append(test_split)
                    for train_metric in metric_types:
                        if (train_metric == "Naturalness"):
                            continue
                        if (train_metric not in all_unordered_res):
                            all_unordered_res[train_metric] = {}

                        for fit_feature_combo in fit_feature_combinations:
                            combo_key = "_".join(fit_feature_combo)

                            for regressor in regression_types:
                                if (regressor + "_" + combo_key not in all_unordered_res[train_metric]):
                                    all_unordered_res[train_metric][regressor + "_" + combo_key] = []

                                regression_model = regression_types[regressor]

                                X = np.concatenate([fit_features[key][train_split] for key in fit_feature_combo], axis=1)
                                y = metric_data[train_metric][train_split]
                                regression_model = regression_model.fit(X, y)

                                test_X = np.concatenate([fit_features[key][test_split] for key in fit_feature_combo], axis=1)
                                pred_y = regression_model.predict(test_X)
                                all_unordered_res[train_metric][regressor + "_" + combo_key].append(pred_y)

                                vbar.update(1)
                
                kf_test_inds = np.concatenate(kf_test_inds)
                inverse_kf_test_inds = np.argsort(kf_test_inds)

                

                for train_metric in all_unordered_res:
                    for regressor in all_unordered_res[train_metric]:
                        all_unordered_res[train_metric][regressor] = np.concatenate(all_unordered_res[train_metric][regressor])[inverse_kf_test_inds]
                        try:
                            res = scistat.pearsonr(all_unordered_res[train_metric][regressor], metric_data[train_metric])
                            p, corr = res.pvalue, res.statistic
                            res_dict[f"Perf/val/{regressor}_{train_metric}/corr"] = corr
                            res_dict[f"Perf/val/{regressor}_{train_metric}/pval"] = p
                            if (corr > best_corr_values[train_metric]):
                                best_corr_keys[train_metric] = f"{regressor}_{train_metric}"
                                best_corr_values[train_metric] = corr
                                save_bools[train_metric] = True
                            if (corr > best_corr):
                                best_corr = corr
                        except:
                            print("Error:", train_metric, regressor, corr_metric)
                vbar.set_description(f"Validation {best_corr}")
                
                with open(Path(config["log_path"]) / config["exp_name"] / "time_stats.json", "w") as fileref:
                    fileref.write(json.dumps(process_times(time_stats), indent=4))
                
                print(best_corr_values)
                print(best_corr_keys)
                
                #if (float(training_config["nll_weight"]) > 0.0):
                #    writer.add_scalar("Loss/val/NLL", text_nll_loss_acum / (batch_i + 1), iters)
                
                for val_key in res_dict:
                    writer.add_scalar(val_key, res_dict[val_key], iters)
                for metric in best_corr_values:
                    writer.add_scalar(f"Perf/val/{metric}_Best/corr", best_corr_values[metric], iters)
    
                model.train()
                text_nll_loss_acum = 0
                valid_ent_loss_acum = 0
                rand_ent_loss_acum = 0
                loss_acum = 0
                
                last_iter = iters
                random.setstate(train_rand_state)
                for metric, save in save_bools.items():
                    if (save):
                        checkpoint_path = Path(config["checkpoint_path"]) / config["exp_name"] / f"best_{metric}_checkpoint.pth"
                        torch.save({
                            "epoch": epoch,
                            "iters": iters + 1,
                            "best_naturalness_corr_value": best_corr_values["Naturalness"],
                            "best_faithfulness_corr_value": best_corr_values["Faithfulness"],
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "scheduler_ca": scheduler_ca.state_dict(),
                        }, checkpoint_path)
                checkpoint_path = Path(config["checkpoint_path"]) / config["exp_name"] / f"latest_checkpoint.pth"
                torch.save({
                    "epoch": epoch,
                    "iters": iters + 1,
                    "best_naturalness_corr_value": best_corr_values["Naturalness"],
                    "best_faithfulness_corr_value": best_corr_values["Faithfulness"],
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_ca": scheduler_ca.state_dict(),
                }, checkpoint_path)
                if (primary_evaluator_model_config["st_emb_cache"] != ""):
                    save_pickle(model.embedding_cache, Path(primary_evaluator_model_config["st_emb_cache"]))
            iters += 1
            
        epoch += 1