from openke.config import Trainer
from openke.module.model import TransH, TransE, TransR, TransD, SimplE, RotatE, DistMult, ComplEx, Analogy, RESCAL
from openke.module.loss import MarginLoss, SoftplusLoss,SigmoidLoss
from openke.module.strategy import NegativeSampling
from openke.data import TrainDataLoader
from pathlib import Path
import numpy as np
import os

def train_transh(
    args,
    dim=300,
    margin=4.0,
    nbatches=100,
    train_times=1000,
    alpha=0.5,
    use_gpu=True,
    ckpt_dir="openke/checkpoint/TransH",
):
    """
    Function to train a TransH model with dynamically generated paths and save its checkpoint.

    Args:
        args (Namespace): Argument namespace containing `data_path`.
        dim (int): Embedding dimension for entities and relations.
        margin (float): Margin for the loss function.
        nbatches (int): Number of batches for training.
        train_times (int): Number of epochs to train the model.
        alpha (float): Learning rate for training.
        use_gpu (bool): Whether to use GPU for training.
        ckpt_dir (str): Directory to save the model checkpoint.
    """
    # Parse `mkg_name` from `args.data_path`
    mkg_name = args.data_path.split('/')[-1].split('.')[0]
    data_path = f"MKGdata/{mkg_name}/"
    ckpt_path = f"{ckpt_dir}/transh_{mkg_name}.ckpt"

    # Data loader for training
    train_dataloader = TrainDataLoader(
        in_path=data_path,
        nbatches=nbatches,
        threads=8,
        sampling_mode="normal",
        bern_flag=1,
        filter_flag=1,
        neg_ent=25,
        neg_rel=0
    )

    # Define the TransH model
    transh = TransH(
        ent_tot=train_dataloader.get_ent_tot(),
        rel_tot=train_dataloader.get_rel_tot(),
        dim=dim,
        p_norm=1,
        norm_flag=True
    )

    # Define the loss function
    model = NegativeSampling(
        model=transh,
        loss=MarginLoss(margin=margin),
        batch_size=train_dataloader.get_batch_size()
    )

    # Train the model
    trainer = Trainer(
        model=model,
        data_loader=train_dataloader,
        train_times=train_times,
        alpha=alpha,
        use_gpu=use_gpu
    )
    trainer.run()

    # Save the trained model checkpoint
    transh.save_checkpoint(ckpt_path)
    print(f"Model checkpoint saved to {ckpt_path}")

    return transh

def train_transr(
    args,
    dim_e=200,
    dim_r=200,
    margin_e=5.0,
    margin_r=4.0,
    nbatches=100,
    train_times_e=1,
    train_times_r=1000,
    alpha_e=0.5,
    alpha_r=1.0,
    use_gpu=True,
    ckpt_dir="openke/checkpoint/TransR",
    result_dir="openke/checkpoint/TransR/result/"
):
    """
    Train a TransR model with a pre-trained TransE model.

    Args:
        args (Namespace): Argument namespace containing `data_path`.
        dim_e (int): Embedding dimension for TransE.
        dim_r (int): Embedding dimension for TransR.
        margin_e (float): Margin for the TransE loss function.
        margin_r (float): Margin for the TransR loss function.
        nbatches (int): Number of batches for training.
        train_times_e (int): Number of epochs to pre-train TransE.
        train_times_r (int): Number of epochs to train TransR.
        alpha_e (float): Learning rate for TransE training.
        alpha_r (float): Learning rate for TransR training.
        use_gpu (bool): Whether to use GPU for training.
        ckpt_dir (str): Directory to save the model checkpoint.
        result_dir (str): Directory to save TransE parameters.

    Returns:
        str: Path to the saved TransR checkpoint.
    """
    # Parse `mkg_name` from `args.data_path`
    mkg_name = args.data_path.split('/')[-1].split('.')[0]
    data_path = f"MKGdata/{mkg_name}/"
    ckpt_path = f"{ckpt_dir}/transr_{mkg_name}.ckpt"
    transe_params_path = f"{result_dir}/transr_transe_{mkg_name}.json"

    # Data loader for training
    train_dataloader = TrainDataLoader(
        in_path=data_path,
        nbatches=nbatches,
        threads=8,
        sampling_mode="normal",
        bern_flag=1,
        filter_flag=1,
        neg_ent=25,
        neg_rel=0
    )

    # Define the TransE model
    transe = TransE(
        ent_tot=train_dataloader.get_ent_tot(),
        rel_tot=train_dataloader.get_rel_tot(),
        dim=dim_e,
        p_norm=1,
        norm_flag=True
    )

    # Define the loss and pre-train TransE
    model_e = NegativeSampling(
        model=transe,
        loss=MarginLoss(margin=margin_e),
        batch_size=train_dataloader.get_batch_size()
    )

    print("Pre-training TransE...")
    trainer = Trainer(
        model=model_e,
        data_loader=train_dataloader,
        train_times=train_times_e,
        alpha=alpha_e,
        use_gpu=use_gpu
    )
    trainer.run()

    # Save TransE parameters
    transe.save_parameters(transe_params_path)
    print(f"TransE parameters saved to {transe_params_path}")

    # Define the TransR model
    transr = TransR(
        ent_tot=train_dataloader.get_ent_tot(),
        rel_tot=train_dataloader.get_rel_tot(),
        dim_e=dim_e,
        dim_r=dim_r,
        p_norm=1,
        norm_flag=True,
        rand_init=False
    )

    # Load pre-trained TransE parameters into TransR
    transr.set_parameters(transe.get_parameters())

    # Define the loss and train TransR
    model_r = NegativeSampling(
        model=transr,
        loss=MarginLoss(margin=margin_r),
        batch_size=train_dataloader.get_batch_size()
    )

    print("Training TransR...")
    trainer = Trainer(
        model=model_r,
        data_loader=train_dataloader,
        train_times=train_times_r,
        alpha=alpha_r,
        use_gpu=use_gpu
    )
    trainer.run()

    # Save the trained TransR model checkpoint
    transr.save_checkpoint(ckpt_path)
    print(f"TransR model checkpoint saved to {ckpt_path}")

    return transr

def train_transd(
    args,
    dim_e=300,
    dim_r=300,
    margin=4.0,
    nbatches=100,
    train_times=1000,
    alpha=1.0,
    use_gpu=True,
    ckpt_dir="openke/checkpoint/TransD",
):
    """
    Train and test a TransD model with dynamically generated paths.

    Args:
        args (Namespace): Argument namespace containing `data_path`.
        dim_e (int): Entity embedding dimension.
        dim_r (int): Relation embedding dimension.
        margin (float): Margin for the loss function.
        nbatches (int): Number of batches for training.
        train_times (int): Number of epochs for training.
        alpha (float): Learning rate for training.
        use_gpu (bool): Whether to use GPU for training and testing.
        ckpt_dir (str): Directory to save the model checkpoint.

    Returns:
        str: Path to the saved TransD checkpoint.
    """
    # Parse `mkg_name` from `args.data_path`
    mkg_name = args.data_path.split('/')[-1].split('.')[0]
    data_path = f"MKGdata/{mkg_name}/"
    ckpt_path = f"{ckpt_dir}/transd_{mkg_name}.ckpt"

    # Data loader for training
    train_dataloader = TrainDataLoader(
        in_path=data_path,
        nbatches=nbatches,
        threads=8,
        sampling_mode="normal",
        bern_flag=1,
        filter_flag=1,
        neg_ent=25,
        neg_rel=0
    )


    # Define the TransD model
    transd = TransD(
        ent_tot=train_dataloader.get_ent_tot(),
        rel_tot=train_dataloader.get_rel_tot(),
        dim_e=dim_e,
        dim_r=dim_r,
        p_norm=1,
        norm_flag=True
    )

    # Define the loss function
    model = NegativeSampling(
        model=transd,
        loss=MarginLoss(margin=margin),
        batch_size=train_dataloader.get_batch_size()
    )

    # Train the model
    print("Training TransD...")
    trainer = Trainer(
        model=model,
        data_loader=train_dataloader,
        train_times=train_times,
        alpha=alpha,
        use_gpu=use_gpu
    )
    trainer.run()

    # Save the trained TransD model checkpoint
    transd.save_checkpoint(ckpt_path)
    print(f"Model checkpoint saved to {ckpt_path}")

    return transd

def train_simple(
    args,
    dim=200,
    nbatches=100,
    train_times=2000,
    alpha=0.5,
    regul_rate=1.0,
    use_gpu=True,
    ckpt_dir="openke/checkpoint/SimpIE",
    opt_method="adagrad"
):
    """
    Train and test a SimplE model with dynamically generated paths.

    Args:
        args (Namespace): Argument namespace containing `data_path`.
        dim (int): Embedding dimension for entities and relations.
        nbatches (int): Number of batches for training.
        train_times (int): Number of epochs for training.
        alpha (float): Learning rate for training.
        regul_rate (float): Regularization rate.
        use_gpu (bool): Whether to use GPU for training and testing.
        ckpt_dir (str): Directory to save the model checkpoint.
        opt_method (str): Optimization method for training.

    Returns:
        str: Path to the saved SimplE checkpoint.
    """
    # Parse `mkg_name` from `args.data_path`
    mkg_name = args.data_path.split('/')[-1].split('.')[0]
    data_path = f"MKGdata/{mkg_name}/"
    ckpt_path = f"{ckpt_dir}/simple_{mkg_name}.ckpt"

    # Data loader for training
    train_dataloader = TrainDataLoader(
        in_path=data_path,
        nbatches=nbatches,
        threads=8,
        sampling_mode="normal",
        bern_flag=1,
        filter_flag=1,
        neg_ent=25,
        neg_rel=0
    )

    # Define the SimplE model
    simple = SimplE(
        ent_tot=train_dataloader.get_ent_tot(),
        rel_tot=train_dataloader.get_rel_tot(),
        dim=dim
    )

    # Define the loss function
    model = NegativeSampling(
        model=simple,
        loss=SoftplusLoss(),
        batch_size=train_dataloader.get_batch_size(),
        regul_rate=regul_rate
    )

    # Train the model
    print("Training SimplE...")
    trainer = Trainer(
        model=model,
        data_loader=train_dataloader,
        train_times=train_times,
        alpha=alpha,
        use_gpu=use_gpu,
        opt_method=opt_method
    )
    trainer.run()

    # Save the trained SimplE model checkpoint
    simple.save_checkpoint(ckpt_path)
    print(f"Model checkpoint saved to {ckpt_path}")

    return simple

def train_rotate(
    args,
    dim=1024,
    margin=6.0,
    epsilon=2.0,
    batch_size=2000,
    train_times=6000,
    alpha=2e-5,
    regul_rate=0.0,
    adv_temperature=2.0,
    use_gpu=True,
    ckpt_dir="./checkpoint/",
    opt_method="adam"
):
    """
    Train and test a RotatE model with dynamically generated paths.

    Args:
        args (Namespace): Argument namespace containing `data_path`.
        dim (int): Embedding dimension for entities and relations.
        margin (float): Margin used in the RotatE model.
        epsilon (float): Epsilon used in the RotatE model.
        batch_size (int): Batch size for training.
        train_times (int): Number of epochs for training.
        alpha (float): Learning rate for training.
        regul_rate (float): Regularization rate for loss function.
        adv_temperature (float): Adversarial temperature for SigmoidLoss.
        use_gpu (bool): Whether to use GPU for training and testing.
        ckpt_dir (str): Directory to save the model checkpoint.
        opt_method (str): Optimization method for training.

    Returns:
        str: Path to the saved RotatE checkpoint.
    """
    # Parse `mkg_name` from `args.data_path`
    mkg_name = args.data_path.split('/')[-1].split('.')[0]
    data_path = f"MKGdata/{mkg_name}/"
    ckpt_path = f"{ckpt_dir}rotate_{mkg_name}.ckpt"

    # Data loader for training
    train_dataloader = TrainDataLoader(
        in_path=data_path,
        batch_size=batch_size,
        threads=8,
        sampling_mode="cross",
        bern_flag=0,
        filter_flag=1,
        neg_ent=64,
        neg_rel=0
    )

    # Define the RotatE model
    rotate = RotatE(
        ent_tot=train_dataloader.get_ent_tot(),
        rel_tot=train_dataloader.get_rel_tot(),
        dim=dim,
        margin=margin,
        epsilon=epsilon
    )

    # Define the loss function
    model = NegativeSampling(
        model=rotate,
        loss=SigmoidLoss(adv_temperature=adv_temperature),
        batch_size=train_dataloader.get_batch_size(),
        regul_rate=regul_rate
    )

    # Train the model
    print("Training RotatE...")
    trainer = Trainer(
        model=model,
        data_loader=train_dataloader,
        train_times=train_times,
        alpha=alpha,
        use_gpu=use_gpu,
        opt_method=opt_method
    )
    trainer.run()

    # Save the trained RotatE model checkpoint
    rotate.save_checkpoint(ckpt_path)
    print(f"Model checkpoint saved to {ckpt_path}")

    return rotate

def train_distmult_adv(
    args,
    dim=1024,
    margin=200.0,
    epsilon=2.0,
    batch_size=2000,
    train_times=400,
    alpha=0.002,
    l3_regul_rate=0.000005,
    adv_temperature=0.5,
    use_gpu=True,
    ckpt_dir="./checkpoint/",
    opt_method="adam"
):
    """
    Train and test a DistMult model with adversarial training.

    Args:
        args (Namespace): Argument namespace containing `data_path`.
        dim (int): Embedding dimension for entities and relations.
        margin (float): Margin used in the DistMult model.
        epsilon (float): Epsilon used in the DistMult model.
        batch_size (int): Batch size for training.
        train_times (int): Number of epochs for training.
        alpha (float): Learning rate for training.
        l3_regul_rate (float): L3 regularization rate.
        adv_temperature (float): Adversarial temperature for SigmoidLoss.
        use_gpu (bool): Whether to use GPU for training and testing.
        ckpt_dir (str): Directory to save the model checkpoint.
        opt_method (str): Optimization method for training.

    Returns:
        str: Path to the saved DistMult checkpoint.
    """
    # Parse `mkg_name` from `args.data_path`
    mkg_name = args.data_path.split('/')[-1].split('.')[0]
    data_path = f"MKGdata/{mkg_name}/"
    ckpt_path = f"{ckpt_dir}distmult_{mkg_name}.ckpt"

    # Data loader for training
    train_dataloader = TrainDataLoader(
        in_path=data_path,
        batch_size=batch_size,
        threads=8,
        sampling_mode="cross",
        bern_flag=0,
        filter_flag=1,
        neg_ent=64,
        neg_rel=0
    )

    # Define the DistMult model
    distmult = DistMult(
        ent_tot=train_dataloader.get_ent_tot(),
        rel_tot=train_dataloader.get_rel_tot(),
        dim=dim,
        margin=margin,
        epsilon=epsilon
    )

    # Define the loss function
    model = NegativeSampling(
        model=distmult,
        loss=SigmoidLoss(adv_temperature=adv_temperature),
        batch_size=train_dataloader.get_batch_size(),
        l3_regul_rate=l3_regul_rate
    )

    # Train the model
    print("Training DistMult with adversarial loss...")
    trainer = Trainer(
        model=model,
        data_loader=train_dataloader,
        train_times=train_times,
        alpha=alpha,
        use_gpu=use_gpu,
        opt_method=opt_method
    )
    trainer.run()

    # Save the trained DistMult model checkpoint
    distmult.save_checkpoint(ckpt_path)
    print(f"Model checkpoint saved to {ckpt_path}")

    return distmult

def train_complex(
    args,
    dim=200,
    nbatches=100,
    train_times=2000,
    alpha=0.5,
    regul_rate=1.0,
    use_gpu=True,
    ckpt_dir="openke/checkpoint/ComplEX",
    opt_method="adagrad"
):
    """
    Train and test a ComplEx model.

    Args:
        args (Namespace): Argument namespace containing `data_path`.
        dim (int): Embedding dimension for entities and relations.
        nbatches (int): Number of batches for training.
        train_times (int): Number of epochs for training.
        alpha (float): Learning rate for training.
        regul_rate (float): Regularization rate.
        use_gpu (bool): Whether to use GPU for training and testing.
        ckpt_dir (str): Directory to save the model checkpoint.
        opt_method (str): Optimization method for training.

    Returns:
        str: Path to the saved ComplEx checkpoint.
    """
    # Parse `mkg_name` from `args.data_path`
    mkg_name = args.data_path.split('/')[-1].split('.')[0]
    data_path = f"MKGdata/{mkg_name}/"
    ckpt_path = f"{ckpt_dir}/complex_{mkg_name}.ckpt"

    # Data loader for training
    train_dataloader = TrainDataLoader(
        in_path=data_path,
        nbatches=nbatches,
        threads=8,
        sampling_mode="normal",
        bern_flag=1,
        filter_flag=1,
        neg_ent=25,
        neg_rel=0
    )

    # Define the ComplEx model
    complEx = ComplEx(
        ent_tot=train_dataloader.get_ent_tot(),
        rel_tot=train_dataloader.get_rel_tot(),
        dim=dim
    )

    # Define the loss function
    model = NegativeSampling(
        model=complEx,
        loss=SoftplusLoss(),
        batch_size=train_dataloader.get_batch_size(),
        regul_rate=regul_rate
    )

    # Train the model
    print("Training ComplEx model...")
    trainer = Trainer(
        model=model,
        data_loader=train_dataloader,
        train_times=train_times,
        alpha=alpha,
        use_gpu=use_gpu,
        opt_method=opt_method
    )
    trainer.run()

    # Save the trained ComplEx model checkpoint
    complEx.save_checkpoint(ckpt_path)
    print(f"Model checkpoint saved to {ckpt_path}")

    return complEx

def train_analogy(
    args,
    dim=200,
    nbatches=100,
    train_times=2000,
    alpha=0.5,
    regul_rate=1.0,
    use_gpu=True,
    ckpt_dir="openke/checkpoint/Analogy",
    opt_method="adagrad"
):
    """
    Train and test an Analogy model.

    Args:
        args (Namespace): Argument namespace containing `data_path`.
        dim (int): Embedding dimension for entities and relations.
        nbatches (int): Number of batches for training.
        train_times (int): Number of epochs for training.
        alpha (float): Learning rate for training.
        regul_rate (float): Regularization rate.
        use_gpu (bool): Whether to use GPU for training and testing.
        ckpt_dir (str): Directory to save the model checkpoint.
        opt_method (str): Optimization method for training.

    Returns:
        str: Path to the saved Analogy checkpoint.
    """
    # Parse `mkg_name` from `args.data_path`
    mkg_name = args.data_path.split('/')[-1].split('.')[0]
    data_path = f"MKGdata/{mkg_name}/"
    ckpt_path = f"{ckpt_dir}/analogy_{mkg_name}.ckpt"

    # Data loader for training
    train_dataloader = TrainDataLoader(
        in_path=data_path,
        nbatches=nbatches,
        threads=8,
        sampling_mode="normal",
        bern_flag=1,
        filter_flag=1,
        neg_ent=25,
        neg_rel=0
    )

    # Define the Analogy model
    analogy = Analogy(
        ent_tot=train_dataloader.get_ent_tot(),
        rel_tot=train_dataloader.get_rel_tot(),
        dim=dim
    )

    # Define the loss function
    model = NegativeSampling(
        model=analogy,
        loss=SoftplusLoss(),
        batch_size=train_dataloader.get_batch_size(),
        regul_rate=regul_rate
    )

    # Train the model
    print("Training Analogy model...")
    trainer = Trainer(
        model=model,
        data_loader=train_dataloader,
        train_times=train_times,
        alpha=alpha,
        use_gpu=use_gpu,
        opt_method=opt_method
    )
    trainer.run()

    # Save the trained Analogy model checkpoint
    analogy.save_checkpoint(ckpt_path)
    print(f"Model checkpoint saved to {ckpt_path}")

    return analogy

def train_rescal(
    args,
    dim=50,
    nbatches=100,
    train_times=1000,
    alpha=0.1,
    margin=1.0,
    use_gpu=True,
    ckpt_dir="openke/checkpoint/RESCAL",
    opt_method="adagrad"
):
    """
    Train and test a RESCAL model.

    Args:
        args (Namespace): Argument namespace containing `data_path`.
        dim (int): Embedding dimension for entities and relations.
        nbatches (int): Number of batches for training.
        train_times (int): Number of epochs for training.
        alpha (float): Learning rate for training.
        margin (float): Margin value for the loss function.
        use_gpu (bool): Whether to use GPU for training and testing.
        ckpt_dir (str): Directory to save the model checkpoint.
        opt_method (str): Optimization method for training.

    Returns:
        str: Path to the saved RESCAL checkpoint.
    """
    # Parse `dataset_name` from `args.data_path`
    dataset_name = args.data_path.split('/')[-1].split('.')[0]
    data_path = f"MKGdata/{dataset_name}/"
    ckpt_path = f"{ckpt_dir}/rescal_{dataset_name}.ckpt"

    # Data loader for training
    train_dataloader = TrainDataLoader(
        in_path=data_path,
        nbatches=nbatches,
        threads=8,
        sampling_mode="normal",
        bern_flag=1,
        filter_flag=1,
        neg_ent=25,
        neg_rel=0
    )

    # Define the RESCAL model
    rescal = RESCAL(
        ent_tot=train_dataloader.get_ent_tot(),
        rel_tot=train_dataloader.get_rel_tot(),
        dim=dim
    )

    # Define the loss function
    model = NegativeSampling(
        model=rescal,
        loss=MarginLoss(margin=margin),
        batch_size=train_dataloader.get_batch_size()
    )

    # Train the model
    print("Training RESCAL model...")
    trainer = Trainer(
        model=model,
        data_loader=train_dataloader,
        train_times=train_times,
        alpha=alpha,
        use_gpu=use_gpu,
        opt_method=opt_method
    )
    trainer.run()

    # Save the trained RESCAL model checkpoint
    rescal.save_checkpoint(ckpt_path)
    print(f"Model checkpoint saved to {ckpt_path}")

    return rescal