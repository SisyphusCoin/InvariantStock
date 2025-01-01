import pandas as pd
import torch
import numpy as np
import random
import os
from dataclasses import dataclass, field
from Layers import *
from tqdm.auto import tqdm

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
@dataclass
class DataArgument:
    save_dir: str = field(
        default='./data',
        metadata={"help": 'directory to save model'}
    )
    start_time: str = field(
        default="2010-12-01",
        metadata={"help": "start_time"}
    )
    end_time: str =field(
        default='2020-12-31', 
        metadata={"help": "end_time"}
    )

    fit_end_time: str= field(
        default="2017-12-31", 
        metadata={"help": "fit_end_time"}
    )

    val_start_time : str = field(
        default='2018-01-01', 
        metadata={"help": "val_start_time"}
    )

    val_end_time: str =field(default='2018-12-31')

    seq_len : int = field(default=20)

    normalize: bool = field(
        default=True,
    )
    select_feature: bool = field(
        default=True,
    )
    use_qlib: bool = field(
        default=False,
    )


def load_model(args):

    feature_extractor = FeatureExtractor(num_latent=args.num_latent, hidden_size=args.hidden_size)
    factor_encoder = FactorEncoder(num_factors=args.num_factor, num_portfolio=args.num_latent, hidden_size=args.hidden_size)
    alpha_layer = AlphaLayer(args.hidden_size)
    beta_layer = BetaLayer(args.hidden_size, args.num_factor)
    factor_decoder = FactorDecoder(alpha_layer, beta_layer)
    factor_prior_model = FatorPrior(args.batch_size, args.hidden_size, args.num_factor)
    predictor = Predictor(feature_extractor, factor_encoder, factor_decoder, factor_prior_model,args)
    
    return predictor

            
@torch.no_grad()
def generate_prediction_scores(masker,model, test_dataloader, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model.to(device)
    masker.to(device)
    model.eval()
    masker.eval()
    ls = []
    with tqdm(total=len(test_dataloader)) as pbar:
        for i, (char, _) in (enumerate(test_dataloader)):
            char = char.float().to(device)
            if char.shape[1] != args.seq_len:
                continue
            char = char[...,:args.feat_dim]
            mask = masker(char.float())[...,0]
            feature = mask * char
            predictions = model.prediction(feature)
            df = pd.DataFrame(predictions.cpu().numpy(), columns=['pred'])
            pbar.update(1)
            ls.append(df)

    return pd.concat(ls,ignore_index=True)


def back_test(output,output_index,dataset,number):
    output["tradable"] = True

    output.index = output_index
    output = pd.merge(dataset,output,left_index=True,right_index=True)
    ## Backtest for China Stock
    # output["tradable"] = ~ ((output["open"] == output["high"]) &  (output["open"] == output["low"]) & (output["open"] == output["close"]))
    # output["tradable"] = output["tradable"].groupby("instrument").shift(-1)
    
    asset = 1
    optimal_asset = 1
    bench_cum = 1
    history = [1]
    optimal_hist = [1]
    bench_hist = [1]
    new_potofolio = potofolio =  new_optimal_potofolio = optimal_potofolio = []
    df_trade_list = []
    profit_list = []
    for date, df in output.groupby("datetime"):
        trade_df =  df.loc[df.tradable].sort_values(by="pred",ascending=False).head(number)
        new_potofolio = trade_df.index.get_level_values("instrument")
        if len(potofolio)==0:
            fee = 0
        else:
            fee = len(set(new_potofolio) - set(potofolio))/len(potofolio) * 0.0015
        profit = trade_df["label"].mean()
        profit_list.append(profit-fee)
        # fee = 0   
        df_trade_list.append(trade_df)
        bench = df["label"].mean()
        pre_asset = asset
        asset *=(1+profit-fee)
        potofolio = new_potofolio
        history.append(asset)
        optimal_hist.append(optimal_asset)
        pre_bench_cum = bench_cum
        bench_cum *=(1+bench)
        bench_hist.append(bench_cum)
    df_history = pd.DataFrame(history[1:],index=all.index.get_level_values("datetime").unique())
    df_history.rename(columns={0:"Cumulative_Returns"},inplace=True)
    df_history['Drawdown'] = df_history['Cumulative_Returns'].div(df_history['Cumulative_Returns'].cummax()) - 1
    df_history["return_rate"] = profit_list
    df_history['Drawdown'].min()

    return df_history

@dataclass
class test_args:
    run_name: str
    num_factor: int
    normalize: bool = True
    select_feature: bool = True
    
    batch_size: int = 300
    seq_length: int = 20

    hidden_size: int = 20
    num_latent: int = 24
    
    save_dir='./best_model'
    use_qlib: bool = False
    device="cuda:0"
    

    
    