import experiment_buddy

experiment_buddy.register(locals())
tensorboard = experiment_buddy.deploy(
    "",
    sweep_yaml="",
    proc_num=1,
    wandb_kwargs={"entity": "ionelia"}
)
