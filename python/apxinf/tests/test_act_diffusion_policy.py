"""Public policy contract: preprocessing, canonical units, and failure behavior."""
import copy
import numpy as np
import pytest
from apxinf.policies.impls.act import ACTPolicy
from apxinf.policies.impls.diffusion import DiffusionPolicy
from apxinf.policies.base import Policy

IMAGE='observation.images.base_0_rgb'
class Runner:
    model_variant='f32'
    def __init__(self, horizon): self.horizon=horizon;self.calls=[]
    def infer_pixels(self,image,state,mean,std,noise=None):
        image=((image.transpose(2,0,1).astype(np.float32)/255-np.asarray(mean,dtype=np.float32)[:,None,None])/np.asarray(std,dtype=np.float32)[:,None,None])[None]
        return self.infer_tensors(image,state,noise)
    def infer_tensors(self,image,state,noise=None):
        self.calls.append((image.copy(),state.copy(),None if noise is None else noise.copy()))
        return np.zeros((self.horizon,2),dtype=np.float32)
    def prepare_tensors(self,mode):return mode


def make(cls):
    config={'type':'act' if cls is ACTPolicy else 'diffusion','input_features':{IMAGE:{'type':'VISUAL','shape':[3,360,640]},'observation.state':{'type':'STATE','shape':[33]}}}
    adapter={'config':{'policy':{'action_space':'lxry'}},'stats':{'norm_stats':{'state':{'q01':[0.]*33,'q99':[2.]*33},'actions':{'mean':[99.,99.],'std':[99.,99.]}},'native_dp_action_min':[-1.,0.],'native_dp_action_max':[1.,.75]}}
    runner=Runner(50 if cls is ACTPolicy else 56)
    return cls(config,adapter,runner),runner,adapter,config

@pytest.mark.parametrize('cls',[ACTPolicy,DiffusionPolicy])
def test_public_protocol_and_preprocessing(cls):
    p,r,_,_=make(cls);assert isinstance(p,Policy)
    obs={IMAGE:np.full((360,640,3),255,dtype=np.uint8),'observation.state':np.ones(33,dtype=np.float32)}
    image=obs[IMAGE].copy();state=obs['observation.state'].copy()
    result=p.infer(obs,noise=None)
    np.testing.assert_allclose(r.calls[-1][0][0,:,0,0],(1-np.array([.485,.456,.406]))/np.array([.229,.224,.225]),rtol=1e-6)
    np.testing.assert_array_equal(r.calls[-1][1],np.zeros((1,33)))
    np.testing.assert_array_equal(obs[IMAGE],image);np.testing.assert_array_equal(obs['observation.state'],state)
    assert result['actions'].shape==(8,2) and result['actions'].dtype==np.float32
    expected=[0.,0.] if cls is ACTPolicy else [0.,.375]
    np.testing.assert_array_equal(result['actions'][0],expected)
    result['actions'][0]=99
    np.testing.assert_array_equal(result['prediction'][0],expected)
    assert p.prepare(obs,mode='graph')=='graph'
    p.close();p.close()
    with pytest.raises(RuntimeError):p.infer(obs)

@pytest.mark.parametrize('cls',[ACTPolicy,DiffusionPolicy])
def test_rejects_invalid_observations_statistics_and_options(cls):
    p,_,adapter,config=make(cls)
    obs={IMAGE:np.zeros((360,640,3),np.uint8),'observation.state':np.zeros(33,np.float32)}
    with pytest.raises(TypeError):p.infer(obs,unsupported=True)
    with pytest.raises(ValueError):p.infer({**obs,IMAGE:obs[IMAGE].astype(np.float32)})
    with pytest.raises(ValueError):p.infer({**obs,'observation.state':np.full(33,np.nan)})
    with pytest.raises(ValueError):p.infer(obs,noise=np.zeros((1,2)))
    bad=copy.deepcopy(adapter);bad['stats']['norm_stats']['state']['q01'][0]=float('nan')
    with pytest.raises(ValueError):cls(config,bad,Runner(50))
    bad=copy.deepcopy(adapter);bad['config']['policy']['action_space']='lowcmd26'
    with pytest.raises(ValueError):cls(config,bad,Runner(50))


def test_diffusion_preserves_exact_noise():
    p,r,_,_=make(DiffusionPolicy)
    noise=np.arange(101*56*2,dtype=np.float32).reshape(101,56,2)
    p.infer({IMAGE:np.zeros((360,640,3),np.uint8),'observation.state':np.zeros(33)},noise=noise)
    np.testing.assert_array_equal(r.calls[0][2],noise)


def test_diffusion_ddpm10_noise_contract():
    _,r,adapter,config=make(DiffusionPolicy)
    config['num_inference_steps']=10
    policy=DiffusionPolicy(config,adapter,r)
    obs={IMAGE:np.zeros((360,640,3),np.uint8),'observation.state':np.zeros(33)}
    noise=np.arange(11*56*2,dtype=np.float32).reshape(11,56,2)
    policy.infer(obs,noise=noise)
    np.testing.assert_array_equal(r.calls[-1][2],noise)
    policy.infer(obs)
    assert r.calls[-1][2].shape==(11,56,2)
    np.testing.assert_array_equal(r.calls[-1][2][-1],np.zeros((56,2)))
    assert policy.metadata['denoising_steps']==10
    with pytest.raises(ValueError):policy.infer(obs,noise=np.zeros((101,56,2),np.float32))
