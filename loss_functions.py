import copy 
import torch
import utils 
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from skimage.measure import label


def pseudo_bce_loss(feat1, output2, anchor, same=False):
    ### Triplet Objective
    anchor = F.normalize(anchor, dim=-1)
    output2 = F.normalize(output2, dim=-1)

    feat_detach = feat1.detach()
    feat_norm = feat_detach / torch.norm(feat_detach, 2, 1, keepdim=True)
    cosine_dist = torch.mm(feat_norm, feat_norm.t())
    vals, pos_idx = torch.topk(cosine_dist, 2, dim=1, largest=True)

    pos_idx = pos_idx[:, 1].cpu().numpy().flatten().tolist()

    positive =  output2[pos_idx, :]
    pos_sim = torch.bmm(anchor.view(anchor.size(0), 1, -1), positive.view(anchor.size(0), -1, 1)).squeeze()
    ones = torch.ones_like(pos_sim)

    bce = nn.BCEWithLogitsLoss()
    loss_bce =  bce(pos_sim, ones)

    return loss_bce


class DINOLoss(nn.Module):
    def __init__(self, out_dim, ncrops, warmup_teacher_temp, teacher_temp,
                 warmup_teacher_temp_epochs, nepochs, student_temp=0.1,
                 surrogate_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.surrogate_momentum = surrogate_momentum
        self.ncrops = ncrops
        self.register_buffer("surrogate", torch.zeros(1, out_dim))
        # we apply a warm up for the teacher temperature because
        # a too high temperature makes the training instable at the beginning
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp,
                        teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))

    def forward(self, student_output, teacher_output, epoch):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        student_out = student_output / self.student_temp
        student_out = student_out.chunk(self.ncrops)

        # teacher surrogateing and sharpening
        temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax((teacher_output - self.surrogate) / temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(2)

        total_loss = 0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    # we skip cases where student and teacher operate on the same view
                    continue
                loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
        total_loss /= n_loss_terms
        self.update_surrogate(teacher_output)
        return total_loss

    @torch.no_grad()
    def update_surrogate(self, teacher_output):
        """
        Update surrogate used for teacher output.
        """
        batch_surrogate = torch.sum(teacher_output, dim=0, keepdim=True)
        dist.all_reduce(batch_surrogate)
        batch_surrogate = batch_surrogate / (len(teacher_output) * dist.get_world_size())

        # ema update
        self.surrogate = self.surrogate * self.surrogate_momentum + batch_surrogate * (1 - self.surrogate_momentum)

class CELoss(nn.Module):
    def __init__(self, nclasses, in_dim, batch_size):
        super(CELoss, self).__init__()
        self.nclasses = nclasses
        self.in_dim = in_dim
        self.batch_size = batch_size
        self.linear = nn.Linear(self.in_dim, self.nclasses,)
        if torch.cuda.is_available():
            self.loss = nn.CrossEntropyLoss().cuda()
        else:
            self.loss = nn.CrossEntropyLoss()
      
    def forward(self, student_input, labels):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        logits = self.linear(student_input.float())
        total_loss = 0

        for crop in range(2): # only interate through two global crops
            idx_start = crop * self.batch_size
            idx_end = idx_start + self.batch_size
            total_loss += self.loss(logits[idx_start:idx_end], labels) / self.batch_size

        total_loss /= 2 # divide two global crops
        return total_loss

class SupervisedContrastiveDINOLoss(nn.Module):
    def __init__(self, out_dim, ncrops, warmup_teacher_temp, teacher_temp,
                 warmup_teacher_temp_epochs, nepochs, batch_size_per_gpu, student_temp=0.1,
                 surrogate_momentum=0.9, use_local_crops = True, multi_layer_loss = False, num_heads = 6, random_crops_number=0):
        super().__init__()
        self.student_temp = student_temp
        self.surrogate_momentum = surrogate_momentum
        self.ncrops = ncrops
        self.register_buffer("surrogate", torch.zeros(1, out_dim))
        # we apply a warm up for the teacher temperature because
        # a too high temperature makes the training instable at the beginning
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp,
                        teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))
        self.batch_size_per_gpu = batch_size_per_gpu
        self.bs_crops = self.batch_size_per_gpu * 1
        self.bs_local_crops = self.batch_size_per_gpu * (self.ncrops - 1)
        self.random_crops_number = random_crops_number
        self.bs_random_crops = self.batch_size_per_gpu * self.random_crops_number 
        self.use_local_crops = use_local_crops
        self.multi_layer_loss = multi_layer_loss
        
        self.student_feat_out = {}
        self.teacher_feat_out = {}
        
        self.num_heads = num_heads 

    def forward(self, student_output, teacher_output, labels, epoch, student_patches = None, teacher_patches = None):

        student_out = student_output / self.student_temp
        student_out = F.log_softmax(student_out, dim=-1)

        # teacher surrogateing and sharpening
        temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax((teacher_output - self.surrogate) / temp, dim=-1, dtype = student_out.dtype).detach()

        extended_labels_teacher = labels#.repeat(1)
        student_crops = int(student_output.shape[0] // self.batch_size_per_gpu)
        extended_labels_student = labels.repeat(student_crops)
        n = self.bs_crops + self.bs_local_crops + self.bs_random_crops if student_crops > 1 else self.bs_crops  

        if not self.use_local_crops:
            same_idx_matrix = (extended_labels_teacher.unsqueeze(1) == extended_labels_teacher.unsqueeze(0)) * 1.0
            if student_crops >1:
                same_idx_matrix = torch.concat([same_idx_matrix, torch.eye(self.batch_size_per_gpu, self.batch_size_per_gpu).repeat(1, self.ncrops - 1 + self.random_crops_number).cuda()], axis = 1)
        else:
            same_idx_matrix = (extended_labels_teacher.unsqueeze(1) == extended_labels_student.unsqueeze(0)) * 1.0
        
        mask_global_id_matrix = torch.concat([torch.eye(self.bs_crops, self.bs_crops), torch.zeros([self.bs_crops, n - self.bs_crops])], axis = 1).cuda()
        valid_idx_matrix = same_idx_matrix * (1 - mask_global_id_matrix)
        valid_idx_count = valid_idx_matrix.sum()
        loss_matrix = -teacher_out.matmul(student_out.T) 
        valid_loss_matrix = loss_matrix * valid_idx_matrix
        total_loss = valid_loss_matrix.sum() / valid_idx_count

        self.update_surrogate(teacher_output)
        return total_loss

    @torch.no_grad()
    def update_surrogate(self, teacher_output):
        """
        Update surrogate used for teacher output.
        """
        batch_surrogate = torch.sum(teacher_output, dim=0, keepdim=True)
        dist.all_reduce(batch_surrogate)
        batch_surrogate = batch_surrogate / (len(teacher_output) * dist.get_world_size())

        # ema update
        self.surrogate = self.surrogate * self.surrogate_momentum + batch_surrogate * (1 - self.surrogate_momentum)

    def student_hook_fn_forward_qkv(self, module, input, output):
        if self.use_local_crops == False:
            if 'qkv' not in self.student_feat_out:
                self.student_feat_out["qkv"] = output
        else:
            if 'qkv_global' not in self.student_feat_out:
                self.student_feat_out["qkv_global"] = output
            else:
                self.student_feat_out["qkv_local"] = output
        
    def teacher_hook_fn_forward_qkv(self, module, input, output):
        self.teacher_feat_out["qkv"] = output
        
    def reset(self, ):
        self.student_feat_out = {}
        self.teacher_feat_out = {}
        
def generate_foreground_mask(attn, gaussianblur_kernel_size=1, blur_sigma=1, foreground_threshold=0.6,
                             remove_component_less_than_pixels=3):
    N, C = attn.shape 
    w = int(np.sqrt(C))

    val, idx = torch.sort(attn)
    val /= torch.sum(val, dim=-1, keepdim=True)
    cumval = torch.cumsum(val, dim=-1)
    
    th_attn = cumval > foreground_threshold
    idx2 = torch.argsort(idx)
    th_attn = torch.gather(th_attn, dim=-1, index=idx2)
    th_attn_original = copy.deepcopy(th_attn)

    for j, th_att in enumerate(th_attn):
        labelled = label(th_att.reshape(w, w).cpu().numpy(), background = 0)
        for k in range(1, np.max(labelled) + 1):
            mask = (labelled == k).reshape(-1)
            if np.sum(mask) <= remove_component_less_than_pixels:
                th_attn[j][mask] = 0
        if th_attn[j].max() == False: 
            th_attn[j] = th_attn_original[j]

    return th_attn


def peso_same_cls_find(cls_tokens, k):
    sim_raw = ( F.normalize(cls_tokens, dim=-1) @ F.normalize(cls_tokens, dim=-1).t() )
    sim_raw = sim_raw - torch.eye(sim_raw.size(0)).cuda() * 100000
    sim = sim_raw.clone().view(-1)
    vals, indices = sim.topk(k=k, dim=0, largest=True, sorted=True)
    target_val = vals[-1]
    same_idx_matrix = (sim_raw >= target_val).float()
    return same_idx_matrix

    

class MICMLoss(nn.Module):
    def __init__(self, out_dim, patch_out_dim, ngcrops, nlcrops, warmup_teacher_temp, 
                 teacher_temp, warmup_teacher_temp2, teacher_temp2, 
                 warmup_teacher_temp_epochs, nepochs, student_temp=0.1, 
                 center_momentum=0.9, center_momentum2=0.9,
                 lambda1=1.0, lambda2=1.0, lambda3=1.0, lambda4=1.0, mim_start_epoch=0,
                 batch_size_per_gpu = None, patch_num_global_crops = None, weighted_pool=False,
                 ):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.center_momentum2 = center_momentum2
        self.ngcrops = ngcrops
        self.nlcrops = nlcrops
        self.ncrops = ngcrops + nlcrops
        self.register_buffer("center", torch.zeros(1, out_dim))
        self.register_buffer("center2", torch.zeros(1, 1, patch_out_dim))
        self.lambda1 = lambda1 # cls
        self.lambda2 = lambda2 # patch
        self.lambda3 = lambda3 # MIM
        self.lambda4 = lambda4 # mae
        self.batch_size_per_gpu = batch_size_per_gpu
        self.patch_num_global_crops = patch_num_global_crops
        self.weighted_pool = weighted_pool
       
        # we apply a warm up for the teacher temperature because 
        # a too high temperature makes the training instable at the beginning 
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp,
                        teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))
        self.teacher_temp2_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp2,
                        teacher_temp2, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp2
        )) if mim_start_epoch == 0 else np.concatenate((
            np.ones(mim_start_epoch) * warmup_teacher_temp2,
            np.linspace(warmup_teacher_temp2,
                        teacher_temp2, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs - mim_start_epoch) * teacher_temp2
        ))
            
    def forward(self, student_output, teacher_output, student_local_cls, student_mask, epoch, labels=None, teacher_backbone=None, raw_imgs = None):
        
        # read student, and teacher features 
        if type(teacher_output[1]) == tuple:
            student_cls_tokens, student_patch_features = student_output[0][:,0,:].detach(), student_output[0][:,1:,:].detach()  
            _, teacher_patch_features = teacher_output[0][:,0,:], teacher_output[0][:,1:,:] 
            student_cls, student_patch = student_output[1] 
            teacher_cls, teacher_patch = teacher_output[1] 
            student_attn = student_output[2].mean(axis = 1)[:, 0, 1:].detach()
            student_attn = student_attn / student_attn.sum(axis = -1, keepdims = True)
            teacher_attn = teacher_output[2].mean(axis = 1)[:, 0, 1:].detach()
            teacher_attn = teacher_attn / teacher_attn.sum(axis = -1, keepdims = True)
        else:
            student_cls, student_patch = student_output[0]
            teacher_cls, teacher_patch = teacher_output[0]
            
        if student_local_cls is not None:
            student_cls = torch.cat([student_cls, student_local_cls])
            
        # [CLS] and patch for global patches
        student_cls = student_cls / self.student_temp 
        student_cls_c = student_cls.chunk(self.ncrops)
        student_patch = student_patch / self.student_temp 
        student_patch_c = student_patch.chunk(self.ngcrops)

        # teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]
        temp2 = self.teacher_temp2_schedule[epoch]
        teacher_cls_c = F.softmax((teacher_cls - self.center) / temp, dim=-1)
        teacher_cls_c = teacher_cls_c.detach().chunk(self.ngcrops)
        teacher_patch_c = F.softmax((teacher_patch - self.center2) / temp2, dim=-1)
        teacher_patch_c = teacher_patch_c.detach().chunk(self.ngcrops)

        total_loss1, n_loss_terms1 = torch.Tensor([0]).cuda(), 1
        total_loss2, n_loss_terms2 = torch.Tensor([0]).cuda(), 1
        total_loss3, n_loss_terms3 = torch.Tensor([0]).cuda(), 1
        total_loss4, n_loss_terms4 = torch.Tensor([0]).cuda(), 1
        

        if labels is not None: 
            same_idx_matrix = ((labels.unsqueeze(1) == labels.unsqueeze(0))*1.0).cuda()
            same_idx_count = same_idx_matrix.sum()
            same_idx_matrix_remove_diagonal = same_idx_matrix - torch.eye(self.batch_size_per_gpu).cuda()
            same_idx_count_remove_diagonal = same_idx_count - self.batch_size_per_gpu        
        
        # [cls loss]: self-supervised contrastive between a global and local views, supervised contrastive between two global views
        if self.lambda1 > 0:   
            for q in range(len(teacher_cls_c)):
                for v in range(len(student_cls_c)):
                    if v != q:
                        if labels is not None and v < self.ngcrops: # supervised contrastive between two different global views 
                            loss_matrix = -teacher_cls_c[q].matmul(F.log_softmax(student_cls_c[v], dim=-1).T.float()) 
                            loss1 = loss_matrix * same_idx_matrix_remove_diagonal
                            total_loss1 += loss1.sum() 
                            n_loss_terms1 += same_idx_count_remove_diagonal 
                        else: # self-supervised contrastive between a global and local view 
                            loss1 = torch.sum(-teacher_cls_c[q] * F.log_softmax(student_cls_c[v], dim=-1), dim=-1)
                            total_loss1 += loss1.sum() #mean()
                            n_loss_terms1 += len(loss1)        
                    if v == q and labels is not None: # supervised contrastive between two same global views 
                        loss_matrix = -teacher_cls_c[q].matmul(F.log_softmax(student_cls_c[v], dim=-1).T.float()) 
                        loss1 = loss_matrix * same_idx_matrix_remove_diagonal
                        total_loss1 += loss1.sum() #/ same_idx_count
                        n_loss_terms1 += same_idx_count_remove_diagonal   

        # [pseduo loss]: unsupervised contrastive. 
        if self.lambda2 > 0: # bce psedo train      
            teacher_cls_c_use = teacher_cls.detach().chunk(self.ngcrops)
            for q in range(len(teacher_cls_c)):
                for v in range(len(student_cls_c)):
                    if v != q:
                        if v < self.ngcrops: # between two different global views 
                            loss2 = pseudo_bce_loss(teacher_cls_c_use[q], student_cls_c[v], teacher_cls_c[q])
                            total_loss2 += loss2.sum()#(loss1.sum() + loss2.sum())  #/ same_idx_count
                            n_loss_terms2 += 1   
                    if v == q: # between two same global views 
                            loss2 = pseudo_bce_loss(teacher_cls_c_use[q], student_cls_c[v],  teacher_cls_c[q])
                            total_loss2 += loss2.sum()
                            n_loss_terms2 += 1     

        
        # [MIM loss]: copied from ibot 
        if self.lambda3 > 0:
            for i in range(self.ngcrops):
                loss3 = torch.sum(-teacher_patch_c[i] * F.log_softmax(student_patch_c[i], dim=-1), dim=-1)
                mask = student_mask[i].flatten(-2, -1)
                loss3 = torch.sum(loss3 * mask.float(), dim=-1) / mask.sum(dim=-1).clamp(min=1.0)
                total_loss3 += loss3.mean()
                n_loss_terms3 += 1       
        ###############################################################################################
        ###############################################################################################
        # [MAE loss]: mae loss
        if self.lambda4 > 0:
            predict_imgs = student_output[-1]
            raw_imgs = torch.cat(raw_imgs, dim=0)
            mask = torch.cat((student_mask[0],student_mask[1]), dim=0)
            masks_pixel = mask.float().repeat_interleave(16, dim=1).repeat_interleave(16, dim=2)
            masks_pixel = torch.unsqueeze(masks_pixel, dim=1).repeat(1,3,1,1)

            total_loss4 = torch.mean((predict_imgs - raw_imgs) ** 2 * masks_pixel) / 0.75
            n_loss_terms4 += 1       
        ###############################################################################################
        ############################################################################################### 
            
        if self.lambda1 > 0:
            total_loss1 = total_loss1 / n_loss_terms1 * self.lambda1
        if self.lambda2 > 0:
            total_loss2 = total_loss2 / n_loss_terms2 * self.lambda2
        if self.lambda3 > 0:
            total_loss3 = total_loss3 / n_loss_terms3 * self.lambda3

        total_loss = dict(cls=total_loss1 , pseudo=total_loss2 , mim=total_loss3, mae=total_loss4, loss=total_loss1 + total_loss2 + total_loss3 + total_loss4)
        self.update_center(teacher_cls, teacher_patch)                  
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_cls, teacher_patch):
        """
        Update center used for teacher output.
        """
        cls_center = torch.sum(teacher_cls, dim=0, keepdim=True)
        dist.all_reduce(cls_center)
        cls_center = cls_center / (len(teacher_cls) * dist.get_world_size())
        self.center = self.center * self.center_momentum + cls_center * (1 - self.center_momentum)

        patch_center = torch.sum(teacher_patch.mean(1), dim=0, keepdim=True)
        dist.all_reduce(patch_center)
        patch_center = patch_center / (len(teacher_patch) * dist.get_world_size())
        self.center2 = self.center2 * self.center_momentum2 + patch_center * (1 - self.center_momentum2)
        
        
        
        
        