import os
from os.path import join as pjoin

# Save models
root_dir = 'D:/DS_Proto(TE)/models'
dir_models = pjoin(root_dir, 'TrainedModels')
dir_perfs = pjoin(root_dir, 'Performances') 
dir_lcs = pjoin(root_dir, 'LossCurves')  

dic_model_paths, dic_perf_paths, dic_lcs_paths = {}, {}, {} 
model_types = ['Stage0', 'Base0', 'Base1', 'Proto0', 'Proto1', 'Proto2', 'Final']

for model_type in model_types:
    dic_model_paths[model_type] = pjoin(dir_models, model_type)
    dic_perf_paths[model_type] = pjoin(dir_perfs, model_type)
    dic_lcs_paths[model_type] = pjoin(dir_lcs, model_type)

for model_type in model_types:
    if not os.path.exists(dic_model_paths[model_type]):
        os.makedirs(dic_model_paths[model_type])
    if not os.path.exists(dic_perf_paths[model_type]):
        os.makedirs(dic_perf_paths[model_type])
    if not os.path.exists(dic_lcs_paths[model_type]):
        os.makedirs(dic_lcs_paths[model_type])


dic_channels = {
    'Capilary_refill_rate': [0,1,59],
    'Diastolic_blood_pressure': [2, 60],
    'Fraction_inspired_oxygen': [3, 61],
    'Glasgow_coma_scale_eye_opening': [4,5,6,7,8,9,10,11, 62],
    'Glasgow_coma_scale_motor_response': [i for i in range(12,24)] + [63],
    'Glasgow_coma_scale_total': [i for i in range(24,37)] + [64],
    'Glasgow_coma_scale_verbal_response': [i for i in range(37,49)] + [65],
    'Glucose': [49, 66],
    'Heart_rate': [50, 67],
    'Height': [51, 68],
    'Mean_blood_pressure': [52, 69],
    'Oxygen_saturation': [53, 70],
    'Respiratory_rate': [54, 71],
    'Systolic_blood_pressure': [55, 72],
    'Temperature': [56, 73],
    'Weight': [57, 74],
    'pH': [58, 75]
}