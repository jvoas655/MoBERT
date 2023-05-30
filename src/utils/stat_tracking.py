from copy import deepcopy
import json
import time

def tic(time_dict, key):
    keys = key.split('/')
    current_dict = time_dict

    for k in keys[:-1]:
        if (k not in current_dict):
            tic(current_dict, k)
        
        current_dict = current_dict[k]["Sub Time Dict"]
    if (keys[-1] not in current_dict):
        current_dict[keys[-1]] = {
            "Time Start": None,
            "Sub Time Dict": {},
            "Time Sum": None
        }
    for par_key in current_dict.keys():
        if (par_key != keys[-1] and current_dict[par_key]["Time Start"] is not None):
            toc(current_dict, par_key)
            #raise ValueError(f"Key {par_key} is not stoped.")
    current_dict[keys[-1]]["Time Start"] = time.time()

def toc(time_dict, key = "", alpha = 0.9):
    keys = key.split('/')
    current_dict = time_dict

    def toc_recursive(time_dict):
        if (time_dict["Time Start"] is not None):
            elapsed_time = time.time() - time_dict["Time Start"]
            if (time_dict["Time Sum"] is None):
                time_dict["Time Sum"] = elapsed_time
            else:
                time_dict["Time Sum"] = (1 - alpha) * elapsed_time + alpha * time_dict["Time Sum"]
            time_dict["Time Start"] = None
        for key in time_dict["Sub Time Dict"]:
            toc_recursive(time_dict["Sub Time Dict"][key])

    if key == "":
        for par_key in current_dict.keys():
            if (current_dict[par_key]["Time Start"] is None):
                continue
            toc_recursive(current_dict[par_key])
    else:
        for k in keys[:-1]:
            if (k not in current_dict):
                raise ValueError(f"Key {k} is not set.")
            for par_key in current_dict.keys():
                if (par_key != k and current_dict[par_key]["Time Start"] is not None):
                    raise ValueError(f"Key {par_key} is not stoped.")
            current_dict = current_dict[k]["Sub Time Dict"]
        for par_key in current_dict.keys():
            if (par_key != keys[-1] and current_dict[par_key]["Time Start"] is not None):
                raise ValueError(f"Key {par_key} is not stoped.")
        toc_recursive(current_dict[keys[-1]])        
    

def process_times(time_dict):
    relative_times_dict = deepcopy(time_dict)
    
    def clean_keys_recursive(dict):
        for key in dict.keys():
            del dict[key]["Time Start"]
            clean_keys_recursive(dict[key]["Sub Time Dict"])

    clean_keys_recursive(relative_times_dict)

    def adjust_times_recursive(dict):
        if (len(dict.keys()) == 0):
            return
        time_sum = 0
        for key in dict.keys():
            time_sum += dict[key]["Time Sum"]
            adjust_times_recursive(dict[key]["Sub Time Dict"])
        for key in dict.keys():
            dict[key]["Time Percent"] = dict[key]["Time Sum"] / time_sum
            del dict[key]["Time Sum"]

    adjust_times_recursive(relative_times_dict)
    
    def total_time_adjust_recursive(dict, perc = 1.0):
        for key in dict.keys():
            dict[key]["Relative Time Percent"] = perc * dict[key]["Time Percent"]
            total_time_adjust_recursive(dict[key]["Sub Time Dict"], dict[key]["Relative Time Percent"])
        
    
    total_time_adjust_recursive(relative_times_dict)

    return relative_times_dict



if __name__ == "__main__":
    # Example usage:
    time_stats = {}
    for i in range(1):
        tic(time_stats, "Entail/Encode Motion")
        time.sleep(0.1)
        tic(time_stats, "Entail/Run Entail")
        time.sleep(0.1)
        tic(time_stats, "Entail/Run Entail/Valid Transformer Forward")
        time.sleep(0.1)
        tic(time_stats, "Entail/Run Entail/Valid Res Aggregate")
        time.sleep(0.1)
        tic(time_stats, "Entail/Run Entail/Valid Pres Head")
        time.sleep(0.1)
        tic(time_stats, "Entail/Run Entail/Valid Loss")
        time.sleep(0.1)
        tic(time_stats, "Entail/Run Entail/Rand Tokenize")
        time.sleep(0.1)
        tic(time_stats, "Entail/Run Entail/Rand ATTN")
        time.sleep(0.1)
        tic(time_stats, "Entail/Run Entail/Rand Transformer Forward")
        time.sleep(0.1)
        tic(time_stats, "Entail/Run Entail/Rand Res Aggregate")
        time.sleep(0.1)
        tic(time_stats, "Entail/Run Entail/Rand Pres Head")
        time.sleep(0.1)
        tic(time_stats, "Entail/Run Entail/Rand Weight Proc")
        time.sleep(0.1)
        tic(time_stats, "Entail/Run Entail/Rand Loss")
        time.sleep(0.1)
        tic(time_stats, "Entail/Opt Step")
        time.sleep(0.1)
        tic(time_stats, "NLL/Encode Motion")
        time.sleep(0.1)
        tic(time_stats, "NLL/Mask Pred")
        time.sleep(0.1)
        tic(time_stats, "NLL/Mask Pred/Tokenize")
        time.sleep(0.1)
        tic(time_stats, "NLL/Mask Pred/ATTN")
        time.sleep(0.1)
        tic(time_stats, "NLL/Mask Pred/Mask Proc")
        time.sleep(0.1)
        tic(time_stats, "NLL/Mask Pred/Transformer Forward")
        time.sleep(0.1)
        tic(time_stats, "NLL/Mask Pred/Res Aggregate")
        time.sleep(0.1)
        tic(time_stats, "NLL/Mask Pred/Pred Head")
        time.sleep(0.1)
        tic(time_stats, "NLL/Opt Step")
        time.sleep(0.1)
        print(json.dumps(time_stats, indent=4))
        toc(time_stats)
        time.sleep(0.1)
    proc_times = process_times(time_stats)
    print(json.dumps(proc_times, indent=4))