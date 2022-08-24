import os
import torch
from vit_pytorch.vit import ViT
import matplotlib.pyplot as plt

from config import *
from agents import *
from envs import *
from vit_pytorch.recorder import Recorder
import pathlib

def get_model(model_path):
    sd = torch.load(model_path)
    env_id = default_config['EnvID']
    env_type = default_config['EnvType']

    if env_type == 'mario':
        env = JoypadSpace(gym_super_mario_bros.make(env_id), COMPLEX_MOVEMENT)
    elif env_type == 'atari':
        env = gym.make(env_id)
    else:
        raise NotImplementedError
    input_size = env.observation_space.shape  # 4
    output_size = env.action_space.n  # 2
    use_cuda = default_config.getboolean('UseGPU')
    use_gae = default_config.getboolean('UseGAE')
    use_noisy_net = default_config.getboolean('UseNoisyNet')
    vf_share_layers=default_config.getboolean('vf_share_layers')
    lam = float(default_config['Lambda'])
    num_worker = 1
    num_step = int(default_config['NumStep'])
    ppo_eps = float(default_config['PPOEps'])
    epoch = int(default_config['Epoch'])
    mini_batch = int(default_config['MiniBatch'])
    batch_size = int(num_step * num_worker / mini_batch)
    learning_rate = float(default_config['LearningRate'])
    entropy_coef = float(default_config['Entropy'])
    gamma = float(default_config['Gamma'])
    clip_grad_norm = float(default_config['ClipGradNorm'])
    entropy_coef = float(default_config['Entropy'])
    # vf_loss_coeff = float(default_config['vf_loss_coeff'])
    eta = float(default_config['ETA'])
    life_done = default_config.getboolean('LifeDone')
    # use_icm = default_config.getboolean('UseICM')
    use_vit = True #default_config.getboolean('UseVIT')

    agent = ICMAgent 

    agent = agent(
            input_size,
            output_size,
            num_worker,
            num_step,
            gamma,
            lam=lam,
            learning_rate=learning_rate,
            ent_coef=entropy_coef,
    #         vf_loss_coeff=vf_loss_coeff,
            clip_grad_norm=clip_grad_norm,
            epoch=epoch,
            batch_size=batch_size,
            ppo_eps=ppo_eps,
            eta=eta,
            use_cuda=use_cuda,
            use_gae=use_gae,
            use_noisy_net=use_noisy_net,
    #         use_icm=use_icm,
            use_vit=use_vit,
            vf_share_layers=vf_share_layers,
            logger=None
        )
    agent.model.load_state_dict(sd)
    model = agent.model
    return model

def get_state(state_path,state_mean_path):
    state = torch.load(state_path)
    state = torch.Tensor(state).to('cuda')
    state = state.float() # [16, 4, 84, 84]
    state = state[0][0]   # [1, 4, 84, 84]
    state = torch.stack([state,state,state,state],axis=0).unsqueeze(0)
    print('state',state.shape)

    state_mean = torch.tensor(torch.load(state_mean_path)).to('cuda').float()
    state_mean = state_mean[0][0] # [1, 4, 84, 84]
    state_mean = torch.stack([state_mean,state_mean,state_mean,state_mean],axis=0).unsqueeze(0)
    print('state_mean',state_mean.shape) 
    return state,state_mean

def visualize_vit(model_path,state_path,state_mean_path,savefig_path):
    model = get_model(model_path)
    state, state_mean = get_state(state_path, state_mean_path)
    v = model.feature
    v = Recorder(v)
    preds, attns  = v(state_mean)

    att_mat = attns[0]
    att_mat = torch.mean(att_mat, dim=1)
    print('att_mat', att_mat.shape) # average across all heads

    residual_att = torch.eye(att_mat.size(1))

    aug_att_mat = att_mat.to('cpu') + residual_att.to('cpu')
    aug_att_mat = aug_att_mat / aug_att_mat.sum(dim=-1).unsqueeze(-1)

    # Recursively multiply the weight matrices (layer)
    joint_attentions = torch.zeros(aug_att_mat.size())
    joint_attentions[0] = aug_att_mat[0]
    for n in range(1, aug_att_mat.size(0)):
        joint_attentions[n] = torch.matmul(aug_att_mat[n], joint_attentions[n-1])
    v = joint_attentions[-1]
    grid_size = int(np.sqrt(aug_att_mat.size(-1)))
    mask = v[0, 1:].reshape(grid_size, grid_size).detach().numpy()

    img = state.cpu()[0][0]
    mask = cv2.resize(mask / mask.max(), img.shape)[..., np.newaxis]
    plt.cla()
    fig, axes = plt.subplots(nrows=2,ncols=1)
    axes[0].imshow(mask.squeeze(-1) * img.numpy() )
    axes[1].imshow(img.numpy())
    plt.savefig(savefig_path)

model_paths = [
    'results/_06252022_233218/models/SuperMarioBros-1-1-v0_global_step_49664000.model',
]
savefig_parent_dir = 'results/_06252022_233218/visualize/'
for model_path in model_paths:
    # model_path = 'results/_06252022_233218/models/SuperMarioBros-1-1-v0_global_step_19456000.model'
    state_dir = 'results/_07072022_093135/state'
    savefig_dir = os.path.join(savefig_parent_dir,model_path.split('/')[-1])
    pathlib.Path(savefig_dir).mkdir(exist_ok=True,parents=True)
    for x in range(0,1000,50):
        state_path = os.path.join(state_dir,'mario_states_{}.pt'.format(x))
        state_mean_path = os.path.join(state_dir,'mario_states_mean_{}.pt'.format(x))
        if not os.path.exists(state_path):
            continue
        savefig_path = os.path.join(savefig_dir,f'x{x}.jpg')
        visualize_vit(model_path,state_path,state_mean_path,savefig_path)
    print(savefig_dir)
    # state_path, state_mean_path = 'mario_states.example.pt','mario_states_mean_example.pt'
    # savefig_path ='mario_vit.jpg'


