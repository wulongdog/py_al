import os
import time
import numpy as np

class plt_save:
    """
    保存训练时 loss,losses,top1,top5 数据
    """
    def __init__(self) -> None:
        self.loss_save = []
        self.losses_save = []
        self.top1_save = []
        self.top5_save = []
    
    def update(self,loss,losses,top1,top5):
        self.loss_save.append(loss)
        self.losses_save.append(losses)
        self.top1_save.append(top1)
        self.top5_save.append(top5)
    
    def save(self,args,data_type):
        print("save data(loss,loss,top1,top5) action...")
        save_time = time.time()
        # 不同beta下不同dataset的不同取样策略不同budget的train/test的(loss,losses,top1,top5)的不同训练的数据保存
        if not os.path.exists("{}/{}/{}/{}".format(args.base_dir,args.plt_dir,args.beta,args.dataset)):
            os.makedirs("{}/{}/{}/{}".format(args.base_dir,args.plt_dir,args.beta,args.dataset))
        np.save("{}/{}/{}/{}/{}_{}_{}_loss_{}.npy".format(
            args.base_dir,args.plt_dir,args.beta,args.dataset,args.name,args.start_budget,data_type,str(save_time))
            ,np.array(self.loss_save))
        np.save("{}/{}/{}/{}/{}_{}_{}_losses_{}.npy".format(
            args.base_dir,args.plt_dir,args.beta,args.dataset,args.name,args.start_budget,data_type,str(save_time))
            ,np.array(self.losses_save))
        np.save("{}/{}/{}/{}/{}_{}_{}_top1_{}.npy".format(
            args.base_dir,args.plt_dir,args.beta,args.dataset,args.name,args.start_budget,data_type,str(save_time))
            ,np.array(self.top1_save))
        np.save("{}/{}/{}/{}/{}_{}_{}_top5_{}.npy".format(
            args.base_dir,args.plt_dir,args.beta,args.dataset,args.name,args.start_budget,data_type,str(save_time))
            ,np.array(self.top5_save))
        
    def __str__(self):
        return "loss:{}  lossess:{}  top1:{}  top5:{}".format(self.loss_save,self.losses_save,self.top1_save,self.top5_save)
