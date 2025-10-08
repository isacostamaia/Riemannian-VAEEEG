import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import random, numpy as np, torch

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)  # or  torch.set_deterministic(True)

seed = 0
os.environ['PYTHONHASHSEED'] = str(seed)
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

def train_m1(m1, train_loader, beta, epoch, loss_dict, optimizer):
    m1.train()
    b_loss, b_recon_m1, b_kl_m1  = 0., 0., 0.
    for i, (x, y, dom_id) in enumerate(train_loader):
        # print("epoch: {}, {}-th batch, dom_id: {}".format(epoch, i, np.unique(dom_id)))
        x = x.to(m1.device)
        optimizer.zero_grad()

        #=== M1 ===
        m1_varparams = m1(x, dom_id, epoch)

        l1, diag1  = m1.unsupervised_loss(m1_varparams, x, dom_id, beta)

        loss = l1
        loss = loss.mean(0) #average over the batch 

        loss.backward()

        optimizer.step()
        b_loss += loss.item()
        b_recon_m1 += diag1["recon"].mean().item()
        b_kl_m1 += diag1["beta*kl"].mean().item()

    loss_dict['train_loss'].append(b_loss / len(train_loader))
    loss_dict['train_recon_m1'].append(b_recon_m1 / len(train_loader))
    loss_dict['train_kl_m1'].append(b_kl_m1 / len(train_loader))

    if epoch % 8 == 0:
        print('====> Epoch: {:03d} Loss: {:.2f} ' \
            '\n                    M1 Recon: {:.2f}, M1 KL: {:.2f}'.format(
                epoch,loss_dict['train_loss'][-1], 
                loss_dict['train_recon_m1'][-1], loss_dict['train_kl_m1'][-1]
                )
            )
        
def train_m1m2(m1, m2, train_loader, beta, alpha, epoch, loss_dict, optimizer):
    m1.train()
    m2.train()
    b_loss, b_recon_m2, b_kl_m2, b_yprior, b_yclass, b_recon_m1, b_kl_m1  = 0., 0., 0., 0., 0., 0., 0.
    for i, (x, y, dom_id) in enumerate(train_loader):
        # print("epoch: {}, {}-th batch, dom_id: {}".format(epoch, i, np.unique(dom_id)))
        x = x.to(m1.device)
        y = y.to(m2.device)
        optimizer.zero_grad()

        #=== M1 ===
        m1_varparams = m1(x, dom_id, epoch)
        #=== M2 ===
        m2_varparams = m2(m1_varparams["z1_mean_m1"], y)

        l1, diag1  = m1.unsupervised_loss(m1_varparams, x, dom_id, beta)
        l2, diag2 = m2.supervised_loss(m1_varparams["z1"], m1_varparams["z1_mean_m1"], y, m2_varparams, alpha)

        loss = l1+l2
        loss = loss.mean(0) #average over the batch 

        loss.backward()

        for name, param in m2.named_parameters():
            if param.grad is not None and "qy_z1_logits" in name:
                print(f"Grad {name}: {param.grad.abs().mean().item():.6f}")

        optimizer.step()
        b_loss += loss.item()
        b_recon_m2 += diag2["recon_loss"].mean().item()
        b_kl_m2 += diag2["kl_z2_loss"].mean().item()
        b_yprior += diag2["yprior_loss"].mean().item()
        b_yclass += diag2["yclass_loss"].mean().item()
        b_recon_m1 += diag1["recon"].mean().item()
        b_kl_m1 += diag1["beta*kl"].mean().item()

    loss_dict['train_loss'].append(b_loss / len(train_loader))
    loss_dict['train_recon_m2'].append(b_recon_m2 / len(train_loader))
    loss_dict['train_kl_m2'].append(b_kl_m2 / len(train_loader))
    loss_dict['train_yprior'].append(b_yprior / len(train_loader))
    loss_dict['train_yclass'].append(b_yclass / len(train_loader))
    loss_dict['train_recon_m1'].append(b_recon_m1 / len(train_loader))
    loss_dict['train_kl_m1'].append(b_kl_m1 / len(train_loader))

    if epoch % 8 == 0:
        print('====> Epoch: {:03d} Loss: {:.2f} ' \
            '\n                    M2 Recon: {:.2f} M2 KL: {:.2f}' \
            '\n                    M2 yprior: {:.2f} M2 yclass: {:.2f}' \
            '\n                    M1 Recon: {:.2f}, M1 KL: {:.2f}'.format(
                epoch,loss_dict['train_loss'][-1], 
                loss_dict['train_recon_m2'][-1], loss_dict['train_kl_m2'][-1],
                loss_dict['train_yprior'][-1], loss_dict['train_yclass'][-1],
                loss_dict['train_recon_m1'][-1], loss_dict['train_kl_m1'][-1]
                )
            )
        
def freeze_m1_train_m2(m1, m2, train_loader, loss_dict, optimizer):
    m1.eval()
    m2.train()
    epoch_loss = 0.0
    for i, (x, y, dom_id) in enumerate(train_loader):
        # print("epoch: {}, {}-th batch, dom_id: {}".format(epoch, i, np.unique(dom_id)))
        x = x.to(m1.device)
        y = y.to(m2.device)
        optimizer.zero_grad()

        #=== M1 ===
        with torch.no_grad():
            z1_mean, z1_logvar = m1.encode(x, dom_id)
        #=== M2 ===
        logits = m2.qy_z1_logits(z1_mean)


        loss = m2.cls_loss_fct(logits, y)
        loss = loss.mean(0)

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

        for name, param in m2.named_parameters():
            if param.grad is not None and "qy_z1_logits" in name:
                print(f"Grad {name}: {param.grad.abs().mean().item():.6f}")


    loss_dict['train_loss'].append(epoch_loss / len(train_loader))

def train_m1m2_CCVAE(m1, m2, train_loader, beta, alpha, epoch, K, loss_dict, optimizer, m1m2_varparams=None):
    
    m1.train()
    m2.train()

    b_loss, b_recon_m2, b_m_pzy_m2, b_qzc_m2, b_yz1_m2, b_class, b_recon_m1, b_kl_m1  = 0., 0., 0., 0., 0., 0., 0., 0.
    for i, (x, y, dom_id) in enumerate(train_loader):

        print("epoch: {}, {}-th batch, dom_id: {}".format(epoch, i, np.unique(dom_id)))

        x = x.to(m1.device)
        y = y.to(m2.device)

        optimizer.zero_grad()    
    
        #forward pass
        m1_varparams = m1(x, dom_id, epoch)
        m2_varparams = m2(m1_varparams["z1_mean_m1"], y, K)

        if m1m2_varparams: #debug
            m1m2_varparams["m1"].append(m1_varparams)
            m1m2_varparams["m2"].append(m2_varparams)

        #loss computation
        l1, diag1 = m1.unsupervised_loss(m1_varparams, x, dom_id, beta)
        l2, diag2 = m2.supervised_loss(m1_varparams["z1"], m1_varparams["z1_mean_m1"], y, K, m2_varparams, alpha=alpha)


        loss = l1 + l2
        loss = loss.mean(0) #average over the batch 
        loss.backward()

        optimizer.step()

        b_loss += loss.item()
        b_recon_m2 += diag2["recon"].mean().item()
        b_m_pzy_m2 += diag2["-log_p_z_y"].mean().item()
        b_qzc_m2 += diag2["log_q_y_zc"].mean().item()
        b_yz1_m2 += diag2["log_q_y_z1"].mean().item()
        b_class += diag2["class_loss"].mean().item()
        b_recon_m1 += diag1["recon"].mean().item()
        b_kl_m1 += diag1["beta*kl"].mean().item()


    loss_dict['train_loss'].append(b_loss / len(train_loader))
    loss_dict['train_recon_m2'].append(b_recon_m2 / len(train_loader))
    loss_dict['train_m_pzy_m2'].append(b_m_pzy_m2 / len(train_loader))
    loss_dict['train_qzc_m2'].append(b_qzc_m2 / len(train_loader))
    loss_dict['train_yz1_m2'].append(b_yz1_m2 / len(train_loader))
    loss_dict['train_class'].append(b_class / len(train_loader))
    loss_dict['train_recon_m1'].append(b_recon_m1 / len(train_loader))
    loss_dict['train_kl_m1'].append(b_kl_m1 / len(train_loader))

    if epoch % 1 == 0:
        print('====> Epoch: {:03d} Loss: {:.2f} ' \
            '\n                    M2 Recon: {:.2f} M2 train_m_pzy: {:.2f} M2 qzc_m2: {:.2f}' \
            '\n                    M2 y_z1: {:.2f} M2 class: {:.2f}' \
            '\n                    M1 Recon: {:.2f}, M1 KL: {:.2f}'.format(
                epoch,loss_dict['train_loss'][-1], 
                loss_dict['train_recon_m2'][-1], loss_dict['train_m_pzy_m2'][-1], loss_dict['train_qzc_m2'][-1],
                loss_dict['train_yz1_m2'][-1], loss_dict['train_class'][-1],
                loss_dict['train_recon_m1'][-1], loss_dict['train_kl_m1'][-1]
                )
            )