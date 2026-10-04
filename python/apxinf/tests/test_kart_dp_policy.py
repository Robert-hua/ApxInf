import numpy as np
import pytest
from apxinf.policies.impls.kart_dp import KartDpPolicy

class Runner:
    def infer_tensors(self,image,state,noise):
        self.inputs=(image.copy(),state.copy(),noise.copy())
        out=np.zeros((24,3),np.float32);out[0]=[-2,2,-1]
        return out
    def prepare_tensors(self,mode):return mode

def policy():
    adapter={'state_mean':[1.]*62,'state_std':[2.]*62,'initial_noise':np.zeros((1,24,3)).tolist(),
             'image_mean':[.485,.456,.406],'image_std':[.229,.224,.225]}
    return KartDpPolicy({'action_axes':['steering','throttle','brake'],'weight_branch':'ema'},adapter,Runner())

def test_window_crop_padding_and_control_ranges():
    p=policy();im=np.zeros((4,240,320,3),np.uint8);im[:,8:232,16:304]=255
    ob={'observation.images.front':im,'observation.state':np.full((4,62),3,np.float32)}
    result=p.infer(ob);x,s,n=p.model_runner.inputs
    assert x.shape==(4,3,224,294)
    np.testing.assert_allclose(x[0,:,0,0],(1-np.array([.485,.456,.406]))/np.array([.229,.224,.225]),rtol=1e-6)
    np.testing.assert_array_equal(x[:,:,:,:3],x[:,:,:,3:4].repeat(3,axis=3))
    np.testing.assert_array_equal(s,np.ones((1,248)))
    np.testing.assert_array_equal(result['actions'][0],[-1,1,0])
    np.testing.assert_array_equal(result['prediction'][0],[-2,2,-1])
    old=result['actions'].copy();p.infer(ob);np.testing.assert_array_equal(result['actions'],old)
    assert p.prepare(ob)=='graph'
    p.close()
    with pytest.raises(RuntimeError,match='closed'):p.infer(ob)

def test_invalid_temporal_profile_noise_and_busy_are_rejected():
    p=policy();ob={'observation.images.front':np.zeros((4,224,288,3),np.uint8),'observation.state':np.zeros((4,62))}
    with pytest.raises(ValueError,match='noise'):p.infer(ob,noise=np.zeros((24,3)))
    bad=dict(ob);bad['observation.state']=np.zeros((62,))
    with pytest.raises(ValueError,match='state'):p.infer(bad)
    bad=dict(ob);bad['observation.images.front']=np.zeros((1,224,288,3),np.uint8)
    with pytest.raises(ValueError,match='four'):p.infer(bad)
    p._lock.acquire()
    try:
        with pytest.raises(RuntimeError,match='busy'):p.infer(ob)
    finally:p._lock.release()
