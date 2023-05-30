import argparse
from collections import defaultdict
from multiprocessing import Pool
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
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor
from sklearn.svm import SVR
import sklearn
from utils.stat_tracking import *
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn import preprocessing

import itertools
from torch.optim.lr_scheduler import ReduceLROnPlateau

def save_pickle(obj, filepath):
    with open(filepath, "wb") as f:
        pickle.dump(obj, f)

def load_pickle(filepath):
    with open(filepath, "rb") as f:
        return pickle.load(f)
    
def generate_fit_feature_combinations(fit_features):
    feature_keys = list(fit_features.keys())
    combinations = []

    for i in range(1, len(feature_keys) + 1):
        for combo in itertools.combinations(feature_keys, i):
            invalid = False
            for key in combo:
                if ("RED" in key and key.replace("RED", "") in combo):
                    invalid = True
                    break
            if (invalid):
                continue
            combinations.append(combo)

    return combinations

        
def parse_arguments():
    parser = argparse.ArgumentParser(description = "Process arguments for initial training of motion chunk encoder")
    parser.add_argument("--device", "-d", default = 0, type = str, help = "Device to utilize (-1 = CPU) (> - 1 = GPU)")
    parser.add_argument("--red_dim", default = 128, type = int, help = "")
    parser.add_argument("--config", "-c", default = Path("configs/base_config.yml"))
    parser.add_argument("--checkpoint", "-ck", default="", help="")
    parser.add_argument("--retrain_autoencoders", action="store_true")
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

def calc_corr(args):
    regressors, regressor_models, regression_params, fold_inds, fit_feature_combinations, fit_features, train_data = args
    
    corrs = {}
    params = {}
    
    for i in range(len(regressors)):
        all_unordered_res = {}
        kf_test_inds = []
        for train_split, test_split in fold_inds:
            kf_test_inds.append(test_split)
            

            for fit_feature_combo in fit_feature_combinations:
                combo_key = "_".join(fit_feature_combo)

                
                if (regressors[i] + "_" + combo_key not in all_unordered_res):
                    all_unordered_res[regressors[i] + "_" + combo_key] = []

                X = np.concatenate([fit_features[key][train_split] for key in fit_feature_combo], axis=1)
                y = train_data[train_split]
                
                regressor_models[i] = regressor_models[i].fit(X, y)

                test_X = np.concatenate([fit_features[key][test_split] for key in fit_feature_combo], axis=1)
                pred_y = regressor_models[i].predict(test_X)
                all_unordered_res[regressors[i] + "_" + combo_key].append(pred_y)
        
        kf_test_inds = np.concatenate(kf_test_inds)
        inverse_kf_test_inds = np.argsort(kf_test_inds)

        

        for regressor in all_unordered_res:
            all_unordered_res[regressor] = np.concatenate(all_unordered_res[regressor])[inverse_kf_test_inds]
            
            res = scistat.pearsonr(all_unordered_res[regressor], train_data)
            p, corr = res.pvalue, res.statistic
            corrs[f"{regressor}"] = corr
            params[f"{regressor}"] = regression_params[i]
    return corrs, params
    

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
    model = MotionTextEvalBERT(
        primary_evaluator_model_config, 
        chunk_encoder_config, 
        tokenizer_and_embedders_config,
        tokenizer_path=Path(Path(args.checkpoint).parent) / "tokenizer.tk",
        load_trained_regressors_path = Path(Path(args.checkpoint).parent))
    print(model)
    #model = torch.compile(model, mode="max-autotune")
    model = model.to(device = device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)

    dtest_judge = JudgementDataset(Path(config["cache_key"]) / "cache_eval", Path("../MotionDataset/evaluation_data/"), Path(config["val_path"]), chunk_encoder_config["chunk_size"], chunk_encoder_config["chunk_overlap"], 1 + (200 // (chunk_encoder_config["chunk_size"] + chunk_encoder_config["chunk_overlap"])), device=device)
    dtest_judge_data = DataLoader(dtest_judge, batch_size=training_config["batch_size"], num_workers=8, shuffle=False, drop_last=False)

    model.eval()
    
    with torch.no_grad():

        all_probs = []
        all_cls = []
        all_sot = []
        all_mot = []
        all_text = []
        all_motion = []
        all_stext = []
        all_smotion = []
        all_pred_embs = {}
        all_res_embs = []
        all_faithfullness = []
        all_naturalness = []
        all_models = []
        
        for batch_i, batch_data in tqdm.tqdm(enumerate(dtest_judge_data), total=len(dtest_judge_data), desc="Generating Judge Embs"):
            batch_motion_chunks = batch_data["motion_chunks"].to(device = device, dtype=torch.float32)
            batch_motion_masks = batch_data["motion_masks"].to(device = device, dtype=torch.float32)
            batch_motion_texts = batch_data["texts"]
            batch_motion_faithfulness = batch_data["faithfulness"]
            batch_motion_naturalness = batch_data["naturalness"]
            batch_motion_model = batch_data["model"]
            
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
            all_res_embs.append(res_embs.detach().cpu().numpy())
            for key in pred_embs:
                if (key not in all_pred_embs):
                    all_pred_embs[key] = []
                all_pred_embs[key].append(pred_embs[key].detach().cpu().numpy())
            all_faithfullness.append(batch_motion_faithfulness.detach().cpu().numpy())
            all_naturalness.append(batch_motion_naturalness.detach().cpu().numpy())
            all_models.extend(batch_motion_model)
        all_probs = np.concatenate(all_probs, axis = 0).reshape(-1, 1)
        all_cls = np.concatenate(all_cls, axis = 0)
        all_sot = np.concatenate(all_sot, axis = 0)
        all_mot = np.concatenate(all_mot, axis = 0)
        all_text = np.concatenate(all_text, axis = 0)
        all_motion = np.concatenate(all_motion, axis = 0)
        all_stext = np.concatenate(all_stext, axis = 0)
        all_smotion = np.concatenate(all_smotion, axis = 0)
        all_res_embs = np.concatenate(all_res_embs, axis = 0)
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
        
        model_splits = {}
        for ind, model in enumerate(all_models):
            if (model not in model_splits):
                model_splits[model] = []
            model_splits[model].append(ind)

        fit_features = {
            #"ALL": all_res_embs.reshape(len(all_cls), -1),
            #"PROB": all_probs,
            "PROB_CLS_MOTIONTEXT": np.concatenate([all_probs, all_cls, all_text, all_motion], axis=1),
            #"CLS_SMOTIONTEXT": np.concatenate([all_cls, all_stext, all_smotion], axis=1),
            #"CLS_MOTIONTEXT_SMOTIONTEXT": np.concatenate([all_cls, all_stext, all_smotion, all_stext, all_smotion], axis=1),
            #"CLS": all_cls, 
            #"MOTIONTEXT": np.concatenate([all_text, all_motion], axis=1), 
            #"MOTSOTSIM": all_motiontext_sim, 
            #"SMOTIONTEXT": np.concatenate([all_stext, all_smotion], axis=1), 
            #"SMOTSOTSIM": all_smotiontext_sim
            }
        #for key in all_pred_embs:   
        #    fit_features[key] = all_pred_embs[key]
            

        metric_data = {
            "Naturalness": all_naturalness, 
            "Faithfulness": all_faithfullness
            }
        import itertools

        def create_regression_types():
            kernels = [
                #'linear', 
                #'poly', 
                'rbf', 
                #'sigmoid'
                ]
            epsilons = [0.3]
            degrees = [3]
            gammas = ['scale']
            coef0_vals = [0.0]
            tol_vals = [1e-8]
            C_vals = [3.68]
            shrinking_vals = [False]

            # Cartestian product of all parameters
            params = list(itertools.product(kernels, epsilons, degrees, gammas, coef0_vals, tol_vals, C_vals, shrinking_vals))
            
            regression_types = {}
            regression_params = {}

            for kernel, epsilon, degree, gamma, coef0, tol, C, shrinking in params:
                key = f"SVR_kernel_{kernel}_epsilon_{epsilon}_degree_{degree}_gamma_{gamma}_coef0_{coef0}_tol_{tol}_C_{C}_shrinking_{shrinking}"
                regression_types[key] = SVR(kernel=kernel, epsilon=epsilon, degree=degree, gamma=gamma, coef0=coef0, tol=tol, C=C, shrinking=shrinking, max_iter=int(1e4))
                regression_params[key] = {"kernel":kernel, "epsilon":epsilon, "degree":degree, "gamma":gamma, "coef0":coef0, "tol":tol, "C":C, "shrinking":shrinking}
                #key = f"ScaledSVR_kernel_{kernel}_epsilon_{epsilon}_degree_{degree}_gamma_{gamma}_coef0_{coef0}_tol_{tol}_C_{C}_shrinking_{shrinking}"
                #regression_types[key] =  make_pipeline(preprocessing.StandardScaler(), SVR(kernel=kernel, epsilon=epsilon, degree=degree, gamma=gamma, coef0=coef0, tol=tol, C=C, shrinking=shrinking, max_iter=int(1e4)))
                #regression_params[key] = {"kernel":kernel, "epsilon":epsilon, "degree":degree, "gamma":gamma, "coef0":coef0, "tol":tol, "C":C, "shrinking":shrinking}

            alphas = [0.12]
            solvers = ["saga"]
            tol_vals = [1e-3]
            max_iters = [6]

            # Cartestian product of all parameters
            params = list(itertools.product(alphas, tol_vals, solvers, max_iters))
            
            #regression_types = {}
            #regression_params = {}

            for alpha, tol_val, solver, max_iter in params:
                key = f"Ridge_alpha_{alpha}_tol_{tol_val}_solver_{solver}_max_iter_{max_iter}"
                regression_types[key] = make_pipeline(preprocessing.StandardScaler(), Ridge(alpha = alpha, tol=tol, max_iter=max_iter, solver=solver))
                regression_params[key] = {"alpha":alpha, "tol":tol, "solver":solver, "max_iter":max_iter}

            #activations = ['relu']
            #tols = [1e-1, 1e-2, 1e-3, 1e-4]

            # Cartestian product of all parameters
            #params = list(itertools.product(activations, tols))

            #regression_types = {}
            #regression_params = {}
            
            #for activation, tol in params:
            #    key = f"MLP_activation_{activation}_tol_{tol}"
            #    regression_types[key] = make_pipeline(preprocessing.StandardScaler(), MLPRegressor(activation = activation, hidden_layer_sizes = [2048, 1024, 256, 64], solver='adam', tol=tol))
            #    regression_params[key] = {"activation":activation, "tol":tol}

            return regression_types, regression_params
        regression_types, regression_paramso = create_regression_types()
        print(len(regression_types))
        

        kf_test_inds = []
        all_unordered_res = {}
        res_dict = {}
        num_fold = 10
        fit_feature_combinations = [combo for combo in generate_fit_feature_combinations(fit_features)]
        best_corr_values = {"Naturalness": -1 * float("inf"), "Faithfulness": -1 * float("inf")}
        best_corr_keys = {"Naturalness": "", "Faithfulness": ""}
        best_mcorr_values = {"Naturalness": -1 * float("inf"), "Faithfulness": -1 * float("inf")}
        best_mcorr_keys = {"Naturalness": "", "Faithfulness": ""}
        vbar = tqdm.tqdm(desc=str(best_corr_values), total= num_fold * len(fit_feature_combinations) * len(regression_types) * len(metric_data))
        
        #for corr_metric in metric_data:
        #    res = scistat.pearsonr(all_probs.flatten(), metric_data[corr_metric])
        #    p, corr = res.pvalue, res.statistic
        #    res_dict[f"Perf/val/prob_{corr_metric}/corr"] = corr
        #    res_dict[f"Perf/val/prob_{corr_metric}/pval"] = p
        #    if (corr > best_corr_values[corr_metric]):
        #        best_corr_keys[corr_metric] = f"prob"
        #        best_corr_values[corr_metric] = corr

        # CLS
        kf = KFold(n_splits = num_fold, shuffle=True, random_state = 42)
        fold_inds = [split for split in kf.split(all_cls, all_faithfullness)]


        num_threads = 1
        thread_split = 1
        
        all_res = {"Sample":{}, "Model":{}}
        param_acum = {}

        

        
        

        
        
        for train_metric in metric_data:
            all_res["Sample"][train_metric] = {}
            all_res["Model"][train_metric] = {}

            corrs = {}
            mcorrs = {}
            params = {}

            train_data = metric_data[train_metric]

            res = scistat.pearsonr(all_probs.flatten(), train_data)
            p, corr = res.pvalue, res.statistic
            corrs[f"PROB"] = corr
            params[f"PROB"] = {}
            m_res = [np.mean(all_probs.flatten()[model_splits[model]]) for model in model_splits]
            m_data = [np.mean(train_data[model_splits[model]]) for model in model_splits]
            mres = scistat.pearsonr(m_res, m_data)
            mp, mcorr = mres.pvalue, mres.statistic
            mcorrs[f"PROB"] = mcorr

            
            mp_args = [(list(), list(), list(), fold_inds, fit_feature_combinations, fit_features, train_data) for i in range(num_threads * thread_split)]
            for c, regressor in enumerate(regression_types):
                regressor_model = regression_types[regressor]
                mp_args[c % (num_threads * thread_split)][0].append(regressor)
                mp_args[c % (num_threads * thread_split)][1].append(regressor_model)
                mp_args[c % (num_threads * thread_split)][2].append(regression_paramso[regressor])

            regressors, regressor_models, regression_params, fold_inds, fit_feature_combinations, fit_features, train_data = mp_args[0]
    
            
            
            for i in range(len(regressors)):
                all_unordered_res = {}
                kf_test_inds = []
                for train_split, test_split in fold_inds:
                    kf_test_inds.append(test_split)
                    

                    for fit_feature_combo in fit_feature_combinations:
                        combo_key = "_".join(fit_feature_combo)

                        
                        if (regressors[i] + "_" + combo_key not in all_unordered_res):
                            all_unordered_res[regressors[i] + "_" + combo_key] = []

                        X = np.concatenate([fit_features[key][train_split] for key in fit_feature_combo], axis=1)
                        y = train_data[train_split]
                        
                        regressor_models[i] = regressor_models[i].fit(X, y)

                        test_X = np.concatenate([fit_features[key][test_split] for key in fit_feature_combo], axis=1)
                        pred_y = regressor_models[i].predict(test_X)
                        all_unordered_res[regressors[i] + "_" + combo_key].append(pred_y)
                        vbar.update(1)
                
                kf_test_inds = np.concatenate(kf_test_inds)
                inverse_kf_test_inds = np.argsort(kf_test_inds)

                

                for regressor in all_unordered_res:
                    all_unordered_res[regressor] = np.concatenate(all_unordered_res[regressor])[inverse_kf_test_inds]
                    
                    res = scistat.pearsonr(all_unordered_res[regressor], train_data)
                    p, corr = res.pvalue, res.statistic
                    corrs[f"{regressor}"] = corr
                    params[f"{regressor}"] = regression_params[i]
                    m_res = [np.mean(all_unordered_res[regressor][model_splits[model]]) for model in model_splits]
                    m_data = [np.mean(train_data[model_splits[model]]) for model in model_splits]
                    mres = scistat.pearsonr(m_res, m_data)
                    mp, mcorr = mres.pvalue, mres.statistic
                    mcorrs[f"{regressor}"] = mcorr
                    all_res["Sample"][train_metric][f"{regressor}"] = corrs[f"{regressor}"]
                    all_res["Model"][train_metric][f"{regressor}"] = mcorrs[f"{regressor}"]
                    for param in params[f"{regressor}"]:
                        if (f"{param}:{params[f'{regressor}'][param]}" not in param_acum):
                            param_acum[f"{param}:{params[f'{regressor}'][param]}"] = []
                        param_acum[f"{param}:{params[f'{regressor}'][param]}"].append(corrs[f'{regressor}'])
                    if (corrs[f'{regressor}'] > best_corr_values[train_metric]):
                        best_corr_values[train_metric] = corrs[f'{regressor}']
                        best_corr_keys[train_metric] = f'{regressor}'
                        vbar.set_description(str(best_corr_values) + " - " + str(best_mcorr_values))
                    if (mcorrs[f'{regressor}'] > best_mcorr_values[train_metric]):
                        best_mcorr_values[train_metric] = mcorrs[f'{regressor}']
                        best_mcorr_keys[train_metric] = f'{regressor}'
                        vbar.set_description(str(best_corr_values) + " - " + str(best_mcorr_values))

            '''
            with Pool(num_threads) as pool:
                for corrs, params in pool.imap_unordered(calc_corr, args):
                    for key in corrs:
                        all_res[train_metric][key] = corrs[key]
                        for param in params[key]:
                            if (f"{param}:{params[key][param]}" not in param_acum):
                                param_acum[f"{param}:{params[key][param]}"] = []
                            param_acum[f"{param}:{params[key][param]}"].append(corrs[key])
                        
                        if (corrs[key] > best_corr_values[train_metric]):
                            best_corr_values[train_metric] = corrs[key]
                            best_corr_keys[train_metric] = key
                            vbar.set_description(str(best_corr_values))

                        vbar.update(1)
            '''
                        
        for param in param_acum:
            param_acum[param] = np.mean(param_acum[param]).item()
        param_acum = dict(sorted(param_acum.items())) 
        all_res["means"] = param_acum
        with open("../model_corrs.json", "w") as fileref:
            fileref.write(json.dumps(all_res, indent=4))
        print(best_corr_values)
        print(best_corr_keys)
        print(best_mcorr_values)
        print(best_mcorr_keys)


        for regressor_name in regression_types:
            for feature_name in fit_features:
                for metric in metric_data:
                    X = fit_features[feature_name]
                    y = metric_data[metric]
                    
                    regression_types[regressor_name] = regression_types[regressor_name].fit(X, y)

                    print(Path(Path(args.checkpoint).parent) / ("_".join([regressor_name.split("_")[0], feature_name, metric])+".obj"))
                    print("Score", regression_types[regressor_name].score(X, y))
                    pred = regression_types[regressor_name].predict(X)
                    res = scistat.pearsonr(y, pred)
                    p, corr = res.pvalue, res.statistic
                    print("Corr", corr)
                    
                    save_pickle(regression_types[regressor_name], Path(Path(args.checkpoint).parent) / ("_".join([regressor_name.split("_")[0], feature_name, metric])+".obj"))


        

        