"""Script to show optuna results."""

import pandas as pd
"""Script to show optuna results."""
from pathlib import Path
import pandas as pd
import optuna

#from optuna.integration.shap import ShapleyImportanceEvaluator

# setup
study_name = "20240209C_Martin_wf2_octofull_7x6_global_ardreg"
experiment_id = 0
sequence_id = 0
study_name = "0_1_2_3_4_5.db"

# definition of path
path_optuna_db = Path("./studies/").joinpath(study_name, f"experiment{experiment_id}",f"sequence{sequence_id}","optuna",optuna_file)

study = optuna.create_study(study_name=study_name, storage='sqlite:///' + optuna_dir + '/' + study_name + '.db', load_if_exists=True)
df = study.trials_dataframe(attrs=("number", "value","user_attrs",  "params", "state"))

best_hp = dict()
for optuna_path in optuna_run_paths:
    

    print('-----------------------------------------------------------')
    optuna_dir = str(optuna_path)
    fold_nr=optuna_dir.split('/fold')[-1]
    study_name = f"main.py_fold_{fold_nr}"
    print('Fold: ', fold_nr)
    print('Optuna dir: ', optuna_dir)
    print('Study name: ', study_name)
    
    
    study = optuna.create_study(direction="maximize", study_name=study_name, storage='sqlite:///' + optuna_dir + '/' + study_name + '.db', load_if_exists=True)
    df = study.trials_dataframe(attrs=("number", "value","user_attrs",  "params", "state"))
    #df['num metabolites']=df.iloc[:,2:119].apply(lambda x: x.sum(), axis=1)

    if len(df['value'].unique())!=1:
        
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        
        ## list failed experiments
        display(df[df['state']=='FAIL'])
        
        # number of trials
        print('Number of completed trials: ',df['number'].max())

        # show best trial
        print(df.loc[df['value'].idxmax()])

        best_s = df.loc[df['value'].idxmax()]
        best_hp[f"outerfold{fold_nr}"]= best_s

        # show 10 best trials
        display(df.sort_values(by='value', ascending=False).head(30))

        fig = optuna.visualization.plot_optimization_history(study)
        fig.show()

        fig = optuna.visualization.plot_slice(study)
        fig.show()
        
        
        #evaluator = ShapleyImportanceEvaluator(seed=0)
        #param_importance_without_inf = evaluator.evaluate(study)
        #data = param_importance_without_inf
        #sorted_d = sorted(data.items(), key=lambda x: x[1])
        #display(list(reversed(sorted_d)))

    
# save best parameters
print('Best results (best_hp) are saved')