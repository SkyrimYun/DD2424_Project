# -*- coding: utf-8 -*-

#The code refer some implementation on https://towardsdatascience.com/a-fastai-pytorch-implementation-of-mixmatch-314bb30d0f99

#Importing fastai will also import numpy, pytorch, etc. 
from fastai.basics import *
#import pandas.util.testing as tm
from fastai.vision import *
from numbers import Integral
import seaborn as sns

!nvidia-smi


#Modified from 
K=2
class MultiTransformLabelList(LabelList):
    def __getitem__(self,idxs:Union[int,np.ndarray])->'LabelList':
        "return a single (x, y) if `idxs` is an integer or a new `LabelList` object if `idxs` is a range."
        idxs = try_int(idxs)
        if isinstance(idxs, Integral):
            if self.item is None: x,y = self.x[idxs],self.y[idxs]
            else:                 x,y = self.item   ,0
            if self.tfms or self.tfmargs:
                #I've changed this line to return a list of augmented images
                x = [x.apply_tfms(self.tfms, **self.tfmargs) for _ in range(K)]
            if hasattr(self, 'tfms_y') and self.tfm_y and self.item is None:
                y = y.apply_tfms(self.tfms_y, **{**self.tfmargs_y, 'do_resolve':False})
            if y is None: y=0
            return x,y
        else: return self.new(self.x[idxs], self.y[idxs])
        
#I'll also need to change the default collate function to accomodate multiple augments
def MixmatchCollate(batch):
    batch = to_data(batch)
    if isinstance(batch[0][0],list):
        batch = [[torch.stack(s[0]),s[1]] for s in batch]
    return torch.utils.data.dataloader.default_collate(batch)

#Grab file path to cifar dataset. Will download data if not present
path = untar_data(URLs.CIFAR)

#Custom ImageList with filter function
class MixMatchImageList(ImageList):
    def filter_train(self,num_items,seed=2343):
        train_idxs = np.array([i for i,o in enumerate(self.items) if Path(o).parts[-3] != "test"])
        valid_idxs = np.array([i for i,o in enumerate(self.items) if Path(o).parts[-3] == "test"])
        np.random.seed(seed)
        keep_idxs = np.random.choice(train_idxs,num_items,replace=False)
        self.items = np.array([o for i,o in enumerate(self.items) if i in np.concatenate([keep_idxs,valid_idxs])])
        return self
    
#Create two databunch objects for the labeled and unlabled images. A fastai databunch is a container for train, validation, and
#test dataloaders which automatically processes transforms and puts the data on the gpu.
data_labeled = (MixMatchImageList.from_folder(path)
                .filter_train(500) #Use 500 labeled images for traning
                .split_by_folder(valid="test") #test on all 10000 images in test set
                .label_from_folder()
                .transform(get_transforms(),size=32)
                #On windows, must set num_workers=0. Otherwise, remove the argument for a potential performance improvement
                .databunch(bs=64,num_workers=0)
                .normalize(cifar_stats))

print(type(data_labeled))

print(data_labeled)

train_set = set(data_labeled.train_ds.x.items)
print(train_set)
src = (ImageList.from_folder(path)
        .filter_by_func(lambda x: x not in train_set)
        .split_by_folder(valid="test"))
print(src)

#Grab file path to cifar dataset. Will download data if not present
path = untar_data(URLs.CIFAR)

#Custom ImageList with filter function
class MixMatchImageList(ImageList):
    def filter_train(self,num_items,seed=2343):
        train_idxs = np.array([i for i,o in enumerate(self.items) if Path(o).parts[-3] != "test"])
        valid_idxs = np.array([i for i,o in enumerate(self.items) if Path(o).parts[-3] == "test"])
        np.random.seed(seed)
        keep_idxs = np.random.choice(train_idxs,num_items,replace=False)
        self.items = np.array([o for i,o in enumerate(self.items) if i in np.concatenate([keep_idxs,valid_idxs])])
        return self
    

data_labeled = (MixMatchImageList.from_folder(path)
                .filter_train(500) #Use 500 labeled images for traning
                .split_by_folder(valid="test") #test on all 10000 images in test set
                .label_from_folder()
                .transform(get_transforms(),size=32)
                #On windows, must set num_workers=0. Otherwise, remove the argument for a potential performance improvement
                .databunch(bs=64,num_workers=0)
                .normalize(cifar_stats))

print(type(data_labeled))

train_set = set(data_labeled.train_ds.x.items)
src = (ImageList.from_folder(path)
        .filter_by_func(lambda x: x not in train_set)
        .split_by_folder(valid="test"))
src.train._label_list = MultiTransformLabelList
data_unlabeled = (src.label_from_folder()
         .transform(get_transforms(),size=32)
         .databunch(bs=128,collate_fn=MixmatchCollate,num_workers=0)
         .normalize(cifar_stats))
print(data_unlabeled)
#Databunch with all 50k images labeled, for baseline
data_full = (ImageList.from_folder(path)
        .split_by_folder(valid="test")
        .label_from_folder()
        .transform(get_transforms(),size=32)
        .databunch(bs=128,num_workers=0)
        .normalize(cifar_stats))
#print(data_full)

"""#### Mixup 

"""

from scipy.stats import beta
x = np.linspace(0.01,0.99, 100)
fig, axes = plt.subplots(1,5,figsize=(36,5))
fig.suptitle(r"$\beta(\alpha,\alpha)$ Distribution",fontsize=16)
alphas = [0.2,0.75,1,10,100]
for a, ax in zip(alphas,axes.flatten()):
    ax.set_title(r"$\alpha={}$".format(a))
    ax.plot(x, beta.pdf(x, a, a))

def mixup(a_x,a_y,b_x,b_y,alpha=0.75):
    l = np.random.beta(alpha,alpha)
    l = max(l,1-l)
    x = l * a_x + (1-l) * b_x
    y = l* a_y + (1-l) * b_y
    return x,y

"""#### Sharpening

"""

def sharpen(p,T=0.5):
    u = p ** (1/T)
    return u / u.sum(dim=1,keepdim=True)

a = torch.softmax(torch.randn(10),0)
fig, axes = plt.subplots(1,3,figsize=(24,5))
fig.suptitle("The effect of sharpening on randomly generated distribution")
sns.barplot(x=np.arange(10),y=a,color="blue",ax=axes[0])
axes[0].set_title("$T = 1.0$")
sns.barplot(x=np.arange(10),y=sharpen(a[None,:],0.5)[0],color="orange",ax=axes[1])
axes[1].set_title("$T = 0.5$")
sns.barplot(x=np.arange(10),y=sharpen(a[None,:],0.1)[0],color="red",ax=axes[2])
axes[2].set_title("$T = 0.1$");



model = models.WideResNet(num_groups=3,N=4,num_classes=10,k=2,start_nf=32)


# loss_x_list = []
# loss_u_list = []
# total_loss = []
# ii = 0
def cross_entropy(pred, soft_targets):
    logsoftmax = nn.LogSoftmax()
    return torch.mean(torch.sum(- soft_targets * logsoftmax(pred), 1))
class MixupLoss(nn.Module):
    
    def forward(self, preds, target, unsort=None, ramp=None, bs=None):
        #global ii
        if unsort is None:
            return F.cross_entropy(preds,target)
        preds = preds[unsort]
        preds_l = preds[:bs]
        preds_ul = preds[bs:]
        preds_l = torch.log_softmax(preds_l,dim=1)
        preds_ul = torch.softmax(preds_ul,dim=1)
        loss_x = -(preds_l * target[:bs]).sum(dim=1).mean()
        #loss_u = F.mse_loss(preds_ul,target[bs:])
        #print(type(target[0]))
        #print(target[bs:])
        # for i in range(bs, len(target)):
        #     target[i] = target[i].long()
        loss_u = cross_entropy(preds_ul,target[bs:])
        self.loss_x = loss_x.item()
        self.loss_u = loss_u.item()
        #loss = loss_x + 100 * ramp * loss_u
        # if ii % 200 == 0:
        #     loss_x_list.append(self.loss_x)
        #     loss_u_list.append(self.loss_u)
        #     total_loss.append(loss)
        #     print("loss_x:", self.loss_x)
        #     print("loss_u:", self.loss_u)
        #     print("total_loss:", loss)
        # ii += 1
        return loss_x + 100 * ramp * loss_u
        #return loss_x + 25 * loss_u
        #return loss



class MixMatchTrainer(LearnerCallback):
    _order=-20
    def on_train_begin(self, **kwargs):
        self.l_dl = iter(data_labeled.train_dl)
        self.smoothL, self.smoothUL = SmoothenValue(0.98), SmoothenValue(0.98)
        self.recorder.add_metric_names(["l_loss","ul_loss"])
        self.it = 0
        
    def on_batch_begin(self, train, last_input, last_target, **kwargs):
        if not train: return
        try:
            x_l,y_l = next(self.l_dl)
        except:
            self.l_dl = iter(data_labeled.train_dl)
            x_l,y_l = next(self.l_dl)
            
        x_ul = last_input
        
        with torch.no_grad():
            ul_labels = sharpen(torch.softmax(torch.stack([self.learn.model(x_ul[:,i]) for i in range(x_ul.shape[1])],dim=1),dim=2).mean(dim=1))
            
        x_ul = torch.cat([x for x in x_ul])
        ul_labels = torch.cat([y.unsqueeze(0).expand(K,-1) for y in ul_labels])
        
        l_labels = torch.eye(data_labeled.c).cuda()[y_l]
        
        w_x = torch.cat([x_l,x_ul])
        w_y = torch.cat([l_labels,ul_labels])
        idxs = torch.randperm(w_x.shape[0])
        
        mixed_input, mixed_target = mixup(w_x,w_y,w_x[idxs],w_y[idxs])
        bn_idxs = torch.randperm(mixed_input.shape[0])
        unsort = [0] * len(bn_idxs)
        for i,j in enumerate(bn_idxs): unsort[j] = i
        mixed_input = mixed_input[bn_idxs]
    

        ramp = self.it / 3000.0 if self.it < 3000 else 1.0
        return {"last_input": mixed_input, "last_target": (mixed_target,unsort,ramp,x_l.shape[0])}
    
    def on_batch_end(self, train, **kwargs):
        if not train: return
        self.smoothL.add_value(self.learn.loss_func.loss_x)
        self.smoothUL.add_value(self.learn.loss_func.loss_u)
        self.it += 1

    def on_epoch_end(self, last_metrics, **kwargs):
        return add_metrics(last_metrics,[self.smoothL.smooth,self.smoothUL.smooth])


learn = Learner(data_unlabeled,model,loss_func=MixupLoss(),callback_fns=[MixMatchTrainer],metrics=accuracy)
#probs, val_labels = learn.get_preds(ds_type=DatasetType.Valid)



learn.recorder.plot()

# loss_x_list = []
# loss_u_list = []
# total_loss = []
# i = 0
learn.fit_one_cycle(200,2e-3,wd=0.02)

# loss_x_list = []
# loss_u_list = []
# total_loss = []

import matplotlib.pyplot as plt
#plot_x = np.linspace(0, 1000, 5000)
plt.plot(loss_x_list, label="x loss")
#plt.plot(acc_val, label="Validation Acc")
plt.xlabel('sample times(every 200 iterations)')
plt.ylabel('x loss')
plt.title('x loss in training set')
plt.legend(loc='best')
plt.savefig("/content/drive/My Drive/Colab Notebooks/project/xloss-1.png")
plt.show()

plt.plot(loss_u_list, label="u loss")
#plt.plot(loss_y_val, label="Validation loss")
plt.xlabel('sample times(every 200 iterations)')
plt.ylabel('u loss')
plt.title('u Loss in training set')
plt.legend(loc='best')
plt.savefig("/content/drive/My Drive/Colab Notebooks/project/uloss-1.png")
plt.show()

print(total_loss[0].item())
total_loss_2 = []
for i in range(len(total_loss)):
    total_loss_2.append(total_loss[i].item())
print(total_loss_2)

train_loss = []
with open("/content/drive/My Drive/Colab Notebooks/project/data.txt", 'r') as file:
    for line in file.readlines():
        line = line.strip('\n')
        train_loss.append(float(line))
print(train_loss)

plt.plot(train_loss, label="u_loss")
#plt.plot(loss_y_val, label="Validation loss")
plt.xlabel('epoch')
plt.ylabel('u_loss')
plt.title('u_loss in validation set')
plt.legend(loc='best')
plt.savefig("/content/drive/My Drive/Colab Notebooks/project/val_u_loss-epoch.png")
plt.show()
