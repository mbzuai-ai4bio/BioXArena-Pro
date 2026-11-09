import os
import json
import numpy as np
import pandas as pd
import pickle

from utils import Ontology
from metrics import compute_metrics

# Setup Paths
EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
GROUND_TRUTH_DIR = os.environ.get("GROUND_TRUTH_DIR", os.path.join(EVAL_DIR, "ground_truth"))
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", os.path.join(EVAL_DIR, "..", "workspace", "output"))

OUTPUT_FILE = "predictions.npy"
INSTANCES = [
    "uniprotkb_swissprot_mfo",
    "uniprotkb_swissprot_bpo",
    "uniprotkb_swissprot_cco",
    "nextprot_mfo",
    "nextprot_bpo"
]
METRIC_NAMES = ["Fmax", "Smin", "AUPR", "AUC"]

class ValidationError(Exception):
    pass

def error_result(error_msg):
    result = {name: None for name in METRIC_NAMES}
    result["error"] = str(error_msg)
    return result

def load_and_validate(instance_name):
    pred_file = os.path.join(OUTPUT_DIR, instance_name, OUTPUT_FILE)
    if not os.path.exists(pred_file):
        raise ValidationError(f"Output file not found: {pred_file}")
    
    try:
        predictions = np.load(pred_file)
    except Exception as e:
        raise ValidationError(f"Failed to load .npy file: {e}")
        
    if not isinstance(predictions, np.ndarray):
        raise ValidationError(f"Output is not a NumPy array. Type: {type(predictions)}")
        
    if predictions.dtype not in [np.float32, np.float64]:
        raise ValidationError(f"Output array must be float32 or float64. Found: {predictions.dtype}")
        
    if np.any((predictions < 0) | (predictions > 1)):
        raise ValidationError("Predictions contain values outside the [0, 1] range.")
        
    if np.isnan(predictions).any():
        raise ValidationError("Predictions contain NaN values.")
        
    return predictions

def calculate_metrics(instance_name, predictions, test_df, go, terms_dict, terms, ont):
    # Ensure shape matches
    expected_shape = (len(test_df), len(terms))
    if predictions.shape != expected_shape:
        raise ValidationError(f"Invalid shape: expected {expected_shape}, got {predictions.shape}")
        
    # Run the exact same logic as deepgo's compute_metrics
    # We ignore avgic, wfmax, wtmax, fmax_spec_match for our task output, but they are returned
    fmax, smin, tmax, wfmax, wtmax, avg_auc, aupr, avgic, fmax_spec_match = compute_metrics(
        test_df, go, terms_dict, terms, ont, predictions
    )
    
    return {
        "Fmax": float(fmax),
        "Smin": float(smin),
        "AUPR": float(aupr),
        "AUC": float(avg_auc)
    }

def run_evaluation():
    results = {}
    
    # Cache ontology structures to avoid redundant loading
    ontologies = {}

    for instance_name in INSTANCES:
        print(f"\n{'='*60}")
        print(f"Evaluating instance: {instance_name}")
        print(f"{'='*60}")
        
        try:
            # Determine subontology from instance name
            if 'mfo' in instance_name:
                ont = 'mf'
            elif 'bpo' in instance_name:
                ont = 'bp'
            elif 'cco' in instance_name:
                ont = 'cc'
            else:
                raise ValueError(f"Unknown subontology for {instance_name}")
                
            predictions = load_and_validate(instance_name)
            
            # Load ground truth files
            gt_dir = os.path.join(GROUND_TRUTH_DIR, instance_name)
            test_df = pd.read_pickle(os.path.join(gt_dir, "y_ref.pkl"))
            terms_df = pd.read_pickle(os.path.join(gt_dir, "terms.pkl"))
            
            # Add exp_annotations if missing in the y_ref? 
            # Wait, test_df has 'exp_annotations'. Yes.
            
            terms = terms_df['gos'].values.flatten()
            terms_dict = {v: i for i, v in enumerate(terms)}
            
            # Load training annotations for IC calculation
            with open(os.path.join(gt_dir, "train_annots.pkl"), "rb") as f:
                train_annots = pickle.load(f)
                
            # Compute IC
            if ont not in ontologies:
                go = Ontology(os.path.join(gt_dir, "go.obo"), with_rels=True)
                ontologies[ont] = go
            else:
                go = ontologies[ont]
                
            test_annotations = test_df['prop_annotations'].values
            test_annotations = list(map(lambda x: set(x), test_annotations))
            go.calculate_ic(list(train_annots) + test_annotations)
            
            scores = calculate_metrics(instance_name, predictions, test_df, go, terms_dict, terms, ont)
            results[instance_name] = scores
            print(f"Results: {scores}")
            
        except ValidationError as e:
            print(f"[Validation Error] {instance_name}: {e}")
            results[instance_name] = error_result(f"Validation: {e}")
        except Exception as e:
            print(f"[Error] {instance_name}: {e}")
            import traceback
            traceback.print_exc()
            results[instance_name] = error_result(e)
            
    return results

if __name__ == "__main__":
    metrics = run_evaluation()
    print("\n=== Final Results ===")
    print(json.dumps(metrics, indent=2))
    with open("score.json", "w") as f:
        json.dump(metrics, f, indent=2)
