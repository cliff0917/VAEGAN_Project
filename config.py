# Train config
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config = {
        'num_shots': 0,
        'device': device,
        'generalized': True,
        'batch_size' : 128,
        'dataset' : 'AWA2',
        'latent_dim': 85,               # size of latent code
        'attr_dim' : 85,                # size of attr
        'nz': 85,                       # size of the latent z vector
        'class_num' : 50,               # number of total classes
        'seen_class_num' : 40,          # number of seen classes
        'unseen_class_num' : 10,        # number of unseen classes
        'd_hdim': 2048,                 # size of the hidden units in discriminator
        'visual_dim': 2048,             # size of the visual feature
        'enc_hidden_dims' : [2048, 1024],   # hidden dims of encoder
        'dec_hidden_dims' : [1024, 2048],   # hidden dims of decoder
        'epochs' : 600,
        'lr': 1e-4,
        'workspace_dir': '.',
        'encoded_noise': True,
        'model_type': 'cvae',
        'n_critic': 1,
    }
