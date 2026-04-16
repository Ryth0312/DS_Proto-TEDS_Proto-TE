#%%
from Configs import *
import torch, time
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import numpy as np

from Utils_basic import *
from Utils_embed import *
from Utils_eval import Evaluate
from tqdm import tqdm

#%%
def ModelInit(model, model_type, device, pretrained_model_name=None):
    ''' Initialize model for different stages of training.'''
    
    if model_type == 'Base0':
        model._init_weights()

    elif model_type == 'Base1':
        print('Loading pretrained `FeatEncoder`&`cls_base0` layers for `Base1` training') 
        # Load Base0 model
        Base0_model_file = pjoin(dic_model_paths['Base0'], pretrained_model_name)
        trained_model = torch.load(Base0_model_file, map_location=device)
        trained_model_dict = trained_model.state_dict()
        
        # Update Base1 model's `FeatEncoder`` & c`ls_base0` layers
        model_dict = model.state_dict()
        update_dict = {}
        for key in trained_model_dict.keys():
            if ('FeatEncoder' in key)|('cls_base0' in key):
                update_dict[key]=trained_model_dict[key]
        model_dict.update(update_dict)
        model.load_state_dict(model_dict)
        model = model.to(device)
        
        # Assert weights are updated
        ## Check whether temporal encoder is updated
        TemporalEncoder_Type = model.TempEnc
        if TemporalEncoder_Type == 'LSTM':
            weight_name = 'FeatEncoder.rnn.weight_ih_l0'
            
        elif TemporalEncoder_Type == 'TF':
            weight_name = 'FeatEncoder.encoder.weight_ih_l0'
            
        elif TemporalEncoder_Type == 'Cat-LSTM':
            weight_name = 'FeatEncoder.encoder_Capilary_refill_rate.weight_ih_l0'
        
        elif TemporalEncoder_Type == 'CW-LSTM':
            weight_name = 'FeatEncoder.encoders.0.weight_ih_l0'
        
        assert model.state_dict()[weight_name].equal(trained_model_dict[weight_name])   
        
        ## Check whether classifier of the temporal encoder is updated
        assert model.state_dict()['cls_base0.weight'].equal(trained_model_dict['cls_base0.weight'])
        
        print('Freeze `FeatEncoder`&`cls_base0` layers in `Base1` training.')
        for key, v in model.named_parameters():
            if ('FeatEncoder' in key)|('cls_base0' in key):
                v.requires_grad = False
            else:
                v.requires_grad = True
        
        # Assert weights are frozen
        if TemporalEncoder_Type == 'LSTM':
            assert model.FeatEncoder.rnn.weight_ih_l0.requires_grad == False
        elif TemporalEncoder_Type == 'TF':
            assert model.FeatEncoder.encoder.weight_ih_l0.requires_grad == False
        elif TemporalEncoder_Type == 'Cat-LSTM':
            assert model.FeatEncoder.encoder_Capilary_refill_rate.weight_ih_l0.requires_grad == False
        elif TemporalEncoder_Type == 'CW-LSTM':
            assert model.FeatEncoder.encoders[0].weight_ih_l0.requires_grad == False
        
        
        assert model.cls_base0.bias.requires_grad == False
        
    
    elif model_type in ['Proto0', 'Proto2']:
        print('Loading pretrained `FeatEncoder` layers for `Proto0` training') 
        # Load Base1 model
        Base1_model_file = pjoin(dic_model_paths['Base1'], pretrained_model_name)
        trained_model = torch.load(Base1_model_file, map_location=device)
        trained_model_dict = trained_model.state_dict()
        
        # Update Base1 model's `FeatEncoder`` & c`ls_base0` layers
        model_dict = model.state_dict()
        update_dict = {}
        for key in trained_model_dict.keys():
            if ('FeatEncoder' in key):
                update_dict[key]=trained_model_dict[key]
        model_dict.update(update_dict)
        model.load_state_dict(model_dict)
        model = model.to(device)
        
        print('Freeze `FeatEncoder` layers in `Base1` training.')
        for key, v in model.named_parameters():
            if ('FeatEncoder' in key):
                v.requires_grad = False
            else:
                v.requires_grad = True
        
        # Assert weights are updated
        print('Check updated & frozen parameters...')
        any_params_trainable = any(param.requires_grad for param in model.FeatEncoder.parameters())
        assert any_params_trainable == False
        
        for k in model.state_dict().keys():
            if k.startswith('FeatEncoder'):
                print(k)
                assert model.state_dict()[k].equal(trained_model_dict[k])
        print('Check Pass!')
    
    elif model_type == 'Proto1':
        print('Loading pretrained `FeatEncoder`&`DemoEncoder` layers from `Base1` model')         
        # Load Base1 model
        Base1_model_file = pjoin(dic_model_paths['Base1'], pretrained_model_name[0])
        trained_model1 = torch.load(Base1_model_file, map_location=device)
        trained_model_dict1 = trained_model1.state_dict()
        
        # Update Base1 model's `FeatEncoder`` & `DemoEncoder` layers
        model_dict = model.state_dict()
        update_dict = {}
        for key in trained_model_dict1.keys():
            if ('FeatEncoder' in key)|('DemoEncoder' in key):
                update_dict[key]=trained_model_dict1[key]
        model_dict.update(update_dict)
        model.load_state_dict(model_dict)
        model = model.to(device)
        
        print('Freeze `FeatEncoder`&`DemoEncoder` layers in `Proto1` training.')
        for key, v in model.named_parameters():
            if ('FeatEncoder' in key)|('DemoEncoder' in key):
                v.requires_grad = False
            else:
                v.requires_grad = True
        
        
        print('Loading pretrained `SynProtoLayer`&`cls_syn_proto` layers from `Proto2` model') 
        # Load Proto2 model
        Proto2_model_file = pjoin(dic_model_paths['Proto2'], pretrained_model_name[1])
        trained_model2 = torch.load(Proto2_model_file, map_location=device)
        trained_model_dict2 = trained_model2.state_dict()
        
        # Update Base1 model's `FeatEncoder`` & `DemoEncoder` layers
        model_dict = model.state_dict()
        update_dict = {}
        for key in trained_model_dict2.keys():
            if ('SynProtoLayer' in key)|('cls_syn_proto' in key):
                update_dict[key]=trained_model_dict2[key]
        model_dict.update(update_dict)
        model.load_state_dict(model_dict)
        model = model.to(device)
        
        print('Freeze `SynProtoLayer`&`cls_syn_proto` layers in `Proto1` training.')
        for key, v in model.named_parameters():
            if ('SynProtoLayer' in key)|('cls_syn_proto' in key):
                v.requires_grad = False
        
        # Assert weights are updated
        print('Check updated & frozen parameters...')
        any_params_trainable1 = any(param.requires_grad for param in model.FeatEncoder.parameters())
        any_params_trainable2 = any(param.requires_grad for param in model.DemoEncoder.parameters())
        any_params_trainable3 = any(param.requires_grad for param in model.SynProtoLayer.parameters())
        any_params_trainable4 = any(param.requires_grad for param in model.cls_syn_proto.parameters())
        assert any_params_trainable1 == False
        assert any_params_trainable2 == False
        assert any_params_trainable3 == False
        assert any_params_trainable4 == False
          
        for k in model.state_dict().keys():
            if (k.startswith('FeatEncoder'))|(k.startswith('DemoEncoder')):
                print(k)
                assert model.state_dict()[k].equal(trained_model_dict1[k])
            elif (k.startswith('SynProtoLayer'))|(k.startswith('cls_syn_proto')):
                print(k)
                assert model.state_dict()[k].equal(trained_model_dict2[k])
        print('Check Pass!')
    
    elif model_type == 'Final':
        print('Loading pretrained `FeatEncoder`&`DemoEncoder`&`SynProtoLayer`&`cls_syn_proto` \
            &`DemoProtoLayer`&`cls_demo_proto` layers for `Proto1` training') 
        # Load Proto1 model
        Proto1_model_file = pjoin(dic_model_paths['Proto1'], pretrained_model_name)
        trained_model = torch.load(Proto1_model_file, map_location=device)
        trained_model_dict = trained_model.state_dict()
        
        # Update Proto1 model's  `FeatEncoder`&`DemoEncoder`
        # &`SynProtoLayer`&`cls_syn_proto`
        # &`DemoProtoLayer`&`cls_demo_proto`  layers
        
        model_dict = model.state_dict()
        update_dict = {}
        for key in trained_model_dict.keys():
            if ('FeatEncoder' in key)|('DemoEncoder' in key)|('SynProtoLayer' in key)|('cls_syn_proto' in key)|('DemoProtoLayer' in key)|('cls_demo_proto' in key):
                update_dict[key]=trained_model_dict[key]
        model_dict.update(update_dict)
        model.load_state_dict(model_dict)
        model = model.to(device)
        
        '''

        print('Freeze `FeatEncoder`&`DemoEncoder`&`SynProtoLayer`&`cls_syn_proto` \
            &`DemoProtoLayer`&`cls_demo_proto` layers in `Proto1` training.')
        for key, v in model.named_parameters():
            if ('FeatEncoder' in key)|('DemoEncoder' in key)|('SynProtoLayer' in key)|('cls_syn_proto' in key)|('DemoProtoLayer' in key)|('cls_demo_proto' in key):
                v.requires_grad = False
            else:
                v.requires_grad = True
        
        # Assert weights are updated
        print('Check updated & frozen parameters...')
        any_params_trainable = any(param.requires_grad for param in model.FeatEncoder.parameters())
        assert any_params_trainable == False
        '''
        
        for k in model.state_dict().keys():
            if (k.startswith('FeatEncoder'))|(k.startswith('DemoEncoder'))|(k.startswith('SynProtoLayer'))|(k.startswith('DemoProtoLayer'))|(k.startswith('cls_syn_proto'))|(k.startswith('cls_demo_proto')):
                print(k)
                assert model.state_dict()[k].equal(trained_model_dict[k])
        print('Check Pass!')
    
      
    # Print trainable weights
    print("Trainable parameters:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)
        
    print('{} Model initialized.'.format(model_type))
    
    return model
    


#%% Model training
#criterion=torch.nn.functional.nll_loss
def Train(model, model_type, optimizer, 
          train_demo_data, train_EMR_data, train_lens, train_labels,
          test_demo_data, test_EMR_data, test_lens, test_labels,
          scheduler=None, epochs=10, proj_inv=4, criterion=torch.nn.functional.nll_loss,
          early_stop=10, batch_size=32, model_file=None, device='cpu', proj_final=False):
    
    opt_loss, opt_roc, flag = 999999, 0, 0
    Train_loss_curve, Test_loss_curve = [], []
    

    train_demo_data,train_EMR_data, train_lens, train_labels = train_demo_data.cpu(), train_EMR_data.cpu(), train_lens.cpu(), train_labels.cpu()
    test_demo_data, test_EMR_data, test_lens, test_labels = test_demo_data.cpu(), test_EMR_data.cpu(), test_lens.cpu(), test_labels.cpu()
    train_Loader = MIMICLoader(train_demo_data, train_EMR_data, train_lens, train_labels, batch_size=batch_size, num_workers=0, shuffle=True)
    for epoch in tqdm(range(epochs)):
        
        # Training
        model.train()
        s_epoch = time.time()
        for batch_idx, (batch_Demo_data, batch_EMR_data, batch_lens, batch_labels) in enumerate(train_Loader):
            batch_Demo_data, batch_EMR_data, batch_lens, batch_labels = batch_Demo_data.to(device), batch_EMR_data.to(device), batch_lens.to(device), batch_labels.to(device)
            
            # Calculate outs & penaltys
            if model_type in ['Base0', 'Base1']:
                out = model(model_type, batch_Demo_data, batch_EMR_data, batch_lens) 
                aux_loss = None
            
            elif model_type == 'Proto0':
                out, dLoss, cLoss, eLoss, L1_loss = model(model_type, batch_Demo_data, batch_EMR_data, batch_lens)
                aux_loss = dLoss + cLoss + eLoss + L1_loss
            
            elif model_type == 'Proto1':
                out, demo_dLoss, syn_dLoss, demo_cLoss, syn_cLoss, demo_eLoss, syn_eLoss, syn_L1_loss, demo_L1_loss = model(model_type, batch_Demo_data, batch_EMR_data, batch_lens)
                dLoss = demo_dLoss #(demo_dLoss + syn_dLoss)
                cLoss = demo_cLoss #(demo_cLoss + syn_cLoss)
                eLoss = demo_eLoss #(demo_eLoss + syn_eLoss)
                L1_loss = demo_L1_loss #(syn_L1_loss + demo_L1_loss)
                aux_loss = dLoss + cLoss + eLoss + L1_loss
            
            elif model_type == 'Proto2':
                out, dLoss, cLoss, eLoss, L1_loss, pLoss = model(model_type, batch_Demo_data, batch_EMR_data, batch_lens)
                aux_loss = dLoss + cLoss + eLoss + L1_loss + pLoss
                
            elif model_type == 'Final':
                out, L1_loss = model(model_type, batch_Demo_data, batch_EMR_data, batch_lens)
                aux_loss = L1_loss
            
            # Calculate accuracy loss
            acc_loss = criterion(out, batch_labels, reduction='sum')

            # Record losses
            if model_type == 'Proto0':
                print('Batch {}/{} Loss:{:.4f} Ld:{:.4f} Lc:{:.4f} Le:{:.4f} L1:{:.4f}'.format(batch_idx, len(train_Loader), acc_loss.detach().cpu().numpy(), 
                                                                                               dLoss.detach().cpu().numpy(), cLoss.detach().cpu().numpy(), eLoss.detach().cpu().numpy(), L1_loss.detach().cpu().numpy()))
            
            elif model_type == 'Proto1':
                print('Batch {}/{} Loss:{:.4f} Ld:{:.4f}|{:.4f} Lc:{:.4f}|{:.4f} Le:{:.4f}|{:.4f} L1:{:.4f}|{:.4f}'.format(batch_idx, len(train_Loader), acc_loss.detach().cpu().numpy(),
                                                                                                                    demo_dLoss.detach().cpu().numpy(), syn_dLoss.detach().cpu().numpy(),
                                                                                                                    demo_cLoss.detach().cpu().numpy(), syn_cLoss.detach().cpu().numpy(),
                                                                                                                    demo_eLoss.detach().cpu().numpy(), syn_eLoss.detach().cpu().numpy(),
                                                                                                                    demo_L1_loss.detach().cpu().numpy(), syn_L1_loss.detach().cpu().numpy()
                                                                                                                    ))
            elif model_type == 'Proto2':
                print('Batch {}/{} Loss:{:.4f} Ld:{:.4f} Lc:{:.4f} Le:{:.4f} L1:{:.4f} Lp:{:.4f}'.format(batch_idx, len(train_Loader), acc_loss.detach().cpu().numpy(),
                                                                                                         dLoss.detach().cpu().numpy(), cLoss.detach().cpu().numpy(), eLoss.detach().cpu().numpy(), L1_loss.detach().cpu().numpy(),
                                                                                                         pLoss.detach().cpu().numpy()))
            
            elif model_type == 'Final':
                print('Batch {}/{} Loss:{:.4f} L1:{:.4f}'.format(batch_idx, len(train_Loader), acc_loss.detach().cpu().numpy(), L1_loss.detach().cpu().numpy()))
                                                                                                        
                                                                                                                 
            
            optimizer.zero_grad()
            if aux_loss is None:
                acc_loss.backward()
            else:
                acc_loss.backward(retain_graph=True)
                aux_loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=5, norm_type=2)
            
            optimizer.step() 
            
            # class weight matrix should be constrined to (0, 1)
            if model_type in ['Proto0', 'Proto2']:
                model.SynProtoLayer.prototypes.data = torch.clamp_(model.SynProtoLayer.prototypes.data, -1, 1)                                                                                            
                #print('Clip prototypes!')
            
            # class weight matrix should be constrined to (0, 1)
            #model.cls_step1.weight.data = torch.clamp_min(model.cls_step1.weight, 0)
            #model.PrototypeLayer.prototypes.data = torch.clamp_(model.PrototypeLayer.prototypes.data, -1, 1)
            #print('Clip class weight maxtrix & prototypes!')
            
        # Learning rate scheduling
        if scheduler is not None:
            print(f'Epoch {epoch}/{epochs}, learning rate: {scheduler.get_last_lr()[0]}')
            #scheduler.step()
            
        
        del batch_Demo_data, batch_EMR_data, batch_lens, batch_labels
        
        # Validation
        model.eval()  
        model.to(device)
        pred_train_out, pred_train_cls = Evaluate(train_demo_data, train_EMR_data, train_lens, train_labels, model, model_type, device=device, batch_size=1024)
        pred_test_out, pred_test_cls = Evaluate(test_demo_data, test_EMR_data, test_lens, test_labels, model, model_type, device=device, batch_size=1024)

        # Loss curves
        train_loss = criterion(pred_train_out.cpu(), train_labels.cpu()).detach().cpu().numpy()
        test_loss = criterion(pred_test_out.cpu(), test_labels.cpu()).detach().cpu().numpy()
        Train_loss_curve.append(train_loss)
        Test_loss_curve.append(test_loss)
        print(f'Epoch {epoch}/{epochs}, train loss: {train_loss:.4f}, test loss: {test_loss:.4f}')
                  
        # Performance 
        pred_train_prob, pred_test_prob = torch.exp(pred_train_out).detach().cpu().numpy(), torch.exp(pred_test_out).detach().cpu().numpy()
        train_metrics = metrics(pred_train_prob[:,1], train_labels.cpu().numpy())
        test_metrics = metrics(pred_test_prob[:,1], test_labels.cpu().numpy())
        print("Train Acc:{:.4f} AUROC:{:.4f} AUPRC:{:.4f}".format(train_metrics[0], train_metrics[1], train_metrics[2]))
        print("Test Acc:{:.4f} AUROC:{:.4f} AUPRC:{:.4f}".format(test_metrics[0], test_metrics[1], test_metrics[2]))
        
        e_epoch = time.time()
        print('Epoch {} costs {}s'.format(epoch, round(e_epoch-s_epoch, 2)))
        
        del pred_train_out, pred_train_cls, pred_test_out, pred_test_cls, pred_train_prob, pred_test_prob
        
        # Early stopping
        test_roc = test_metrics[1]
        
        if test_loss < opt_loss:
            flag = 0
            opt_loss = test_loss
            if model_file == None:
                pass
            else:
                torch.save(model, model_file)
                print('Save model as {}'.format(model_file))
        else:
            flag += 1
            if flag == early_stop:
                print('Early stopping!')
                break
        
        
        # Prototype projection 
        model.eval()
        if model_type.startswith('Proto'):
            if (epoch+1) % proj_inv == 0:
                if model_type in ['Proto0', 'Proto2']:
                    proj_Loader = MIMICLoader(train_demo_data, train_EMR_data, train_lens, train_labels, batch_size=256, num_workers=0, shuffle=False)
                    proto_dist_list, new_prototypes = model.SynProtoLayer.projection(model, proj_Loader, proj_final)
                    
                    '''
                    train_all_ht, train_final_embed = torch.tensor([]).to(device), torch.tensor([]).to(device)
                    for batch_Demo_data, batch_EMR_data, batch_lens, batch_labels in proj_Loader:
                        batch_Demo_data, batch_EMR_data, batch_lens = batch_Demo_data.to(device), batch_EMR_data.to(device), batch_lens.to(device)
                        batch_all_ht, batch_embeddings = model.FeatEncoder(batch_EMR_data, batch_lens)
                        if proj_final:
                            train_final_embed = torch.cat((train_final_embed, batch_embeddings), dim=0)
                            print('Final embeddings:', train_final_embed.shape)
                            assert len(train_all_ht) == 0
                        else:
                            train_all_ht = torch.cat((train_all_ht, batch_all_ht), dim=0)
                            print('All embeddings:', train_all_ht.shape)
                            assert len(train_final_embed) == 0
                    
                    prev_prototypes = model.state_dict()['SynProtoLayer.prototypes']
                    if proj_final:
                        proto_dist_list, new_prototypes = model.SynProtoLayer.projection(train_final_embed)
                    else:
                        proto_dist_list, new_prototypes = model.SynProtoLayer.projection(train_all_ht)
                    
                    
                    print("Prototype projection at epoch {}.".format(epoch+1), '\n', prev_prototypes, '\n', new_prototypes)
                    '''
                elif model_type == 'Proto1':
                    print('Demographic projection')
                    demo_embeddings = F.tanh(model.DemoEncoder(train_demo_data.to(device)))
                    proto_dist_list, new_prototypes = model.DemoProtoLayer.projection_prev(demo_embeddings)
                    
                    
    del train_Loader
 
    return Train_loss_curve, Test_loss_curve

