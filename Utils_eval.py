import os, time
from tqdm import tqdm
from Utils_embed import *


def Evaluate(demo_data, EMR_data, lens, labels, model, model_type, device='cpu', batch_size=512):
    
    s_val = time.time()
    demo_data, EMR_data, lens, labels = demo_data.to('cpu'), EMR_data.to('cpu'), lens.to('cpu'), labels.to('cpu')
    eval_loader = MIMICLoader(demo_data, EMR_data, lens, labels, batch_size=batch_size, shuffle=False, num_workers=0)
    
    pred_out, pred_cls = torch.tensor([]), torch.tensor([])
    
    
    for batch_idx,  (tmp_demo_x, tmp_EMR_x, tmp_lens, tmp_labels) in enumerate(eval_loader):
        tmp_demo_x, tmp_EMR_x, tmp_lens, tmp_labels = tmp_demo_x.to(device), tmp_EMR_x.to(device), tmp_lens.to(device), tmp_labels.to(device)
        
        if model_type in ['Base0', 'Base1']:
            tmp_pred_out = model(model_type, tmp_demo_x, tmp_EMR_x, tmp_lens).cpu()
        elif model_type in ['Proto0', 'Proto1','Proto2']:
            if model_type == 'Proto2':
                # Do not calculate consistency loss
                tmp_pred_out = model('Proto0', tmp_demo_x, tmp_EMR_x, tmp_lens)[0].cpu()
            else:
                tmp_pred_out = model(model_type, tmp_demo_x, tmp_EMR_x, tmp_lens)[0].cpu()
        elif model_type == 'Final':
            tmp_pred_out = model(model_type, tmp_demo_x, tmp_EMR_x, tmp_lens)[0].cpu()
            
        tmp_pred_cls = tmp_pred_out.max(dim=1)[1]
        
        pred_out = torch.cat((pred_out, tmp_pred_out), dim=0)
        pred_cls = torch.cat((pred_cls, tmp_pred_cls), dim=0)
        
    e_val = time.time()
    print('Evaluation costs {}s.'.format(round(e_val-s_val, 2)))
    return pred_out, pred_cls




def LoaderEval(data_loader, model, stage, device='cpu'):
    s_val = time.time()
    pred_out, pred_cls = torch.tensor([]), torch.tensor([])
    #Generate validation results
    for tmp_embs, tmp_lens, tmp_labels, _ in tqdm(data_loader):
        tmp_embs, tmp_lens, tmp_labels = tmp_embs.to(device), tmp_lens.to(device), tmp_labels.to(device)
        # Generate predictions
        if stage == 0:
            tmp_out = model(tmp_embs, tmp_lens, stage=stage).cpu()
            tmp_cls = tmp_out.max(dim=1)[1]
        else:
            tmp_out = model(tmp_embs, tmp_lens, stage=stage, consistency=False)[0].cpu()
            tmp_cls = tmp_out.max(dim=1)[1]
        
        pred_out = torch.cat((pred_out, tmp_out))
        pred_cls = torch.cat((pred_cls, tmp_cls))
    
    e_val = time.time()
    print('Evaluation costs {}s.'.format(round(e_val-s_val, 2)))
    return pred_out, pred_cls