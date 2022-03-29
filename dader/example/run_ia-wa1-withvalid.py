import sys
sys.path.append("..")
import data, model
import torch 
torch.cuda.set_device(1)

# load data
# X_src, y_src = data.load_data(path='split_ia.csv', use_attri=["Song_Name","Artist_Name","Album_Name","Genre","Price","CopyRight","Time","Released"])
X_src, y_src = data.load_data(path='source.csv')
X_tgt, X_tgt_val, y_tgt, y_tgt_val = data.load_data(path='target.csv', valid_rate = 0.1)

# define the model
aligner = model.Model(method = 'invgankd', architecture = 'Bert')
aligner.fit(X_src, y_src, X_tgt, X_tgt_val, y_tgt_val, batch_size = 16, ada_max_epoch=20)                    
y_prd = aligner.predict(X_tgt)
eval_result = aligner.eval(X_tgt, y_prd, y_tgt)

aligner.finetune(X_tgt, y_tgt, X_tgt, X_tgt_val, y_tgt_val)
y_prd = aligner.predict(X_tgt)
eval_result = aligner.eval(X_tgt, y_prd, y_tgt)
print(eval_result)
