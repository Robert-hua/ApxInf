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

def test_explicit_external_vision_uses_raw_crop_and_owned_feature_handoff():
    p=policy()
    class Vision:
        def infer(self,x):
            self.input=x.copy()
            return np.arange(24576,dtype=np.float32).reshape(1,-1)
        def close(self):self.closed=True
        def prepare(self,mode):self.execution=mode
    vision=Vision();p.vision=vision
    image=np.zeros((4,240,320,3),np.uint8);image[:,8:232,16:304]=255
    p.infer({'observation.images.front':image,'observation.state':np.ones((4,62))})
    assert vision.input.shape==(1,4,3,224,288)
    np.testing.assert_array_equal(vision.input,np.ones_like(vision.input))
    features,state,_=p.model_runner.inputs
    np.testing.assert_array_equal(features,np.arange(24576,dtype=np.float32).reshape(1,-1))
    np.testing.assert_array_equal(state,np.zeros((1,248)))
    ob={'observation.images.front':image,'observation.state':np.ones((4,62))}
    assert p.prepare(ob,mode='graph')=='graph';assert vision.execution=='graph'
    assert p.prepare(ob,mode='eager')=='eager';assert vision.execution=='eager'
    p.close();assert vision.closed

@pytest.mark.parametrize('field', ['input_contract', 'engine_sha256', 'weights_sha256'])
def test_external_vision_rejects_unpaired_artifacts_before_cuda(tmp_path, monkeypatch, field):
    import hashlib,json,sys,types
    from apxinf.policies.impls.kart_dp_trt import KartTrtVision
    # No CUDA or TensorRT entrypoint may be reached for mismatched identities.
    monkeypatch.setitem(sys.modules,'tensorrt',types.SimpleNamespace())
    engine=tmp_path/'vision.engine';engine.write_bytes(b'engine-test-fixture')
    weights=tmp_path/'model.safetensors';weights.write_bytes(b'weight-test-fixture')
    metadata={'input_contract':'kart_dp_vision_features_v1',
              'engine_sha256':hashlib.sha256(engine.read_bytes()).hexdigest(),
              'weights_sha256':hashlib.sha256(weights.read_bytes()).hexdigest()}
    metadata[field]='mismatch'
    engine.with_suffix('.json').write_text(json.dumps(metadata))
    with pytest.raises(ValueError,match='identity mismatch'):
        KartTrtVision(engine,tmp_path)

def test_fast_path_uploads_all_four_new_rgb_frames_and_noise():
    p=policy();p._gpu_rgb=True
    class PixelRunner(Runner):
        def infer_pixels(self,image,state,mean,std,noise):
            self.pixel_input=image.copy();self.normalization=(mean,std)
            return self.infer_tensors(image,state,noise)
    p.model_runner=PixelRunner()
    image=np.arange(4*240*320*3,dtype=np.uint8).reshape(4,240,320,3)
    ob={'observation.images.front':image,'observation.state':np.ones((4,62))}
    p.infer(ob);np.testing.assert_array_equal(p.model_runner.pixel_input,image[:,8:232,16:304])
    assert p.model_runner.pixel_input.dtype==np.uint8
    assert p.model_runner.pixel_input.flags.c_contiguous
    changed=np.bitwise_xor(image,np.uint8(255));ob['observation.images.front']=changed
    noise=np.ones((1,24,3),np.float32);p.infer(ob,noise=noise)
    np.testing.assert_array_equal(p.model_runner.pixel_input,changed[:,8:232,16:304])
    np.testing.assert_array_equal(p.model_runner.inputs[2],noise)
    np.testing.assert_allclose(p.model_runner.normalization[0],[.485,.456,.406])

def test_1004_profile_keeps_full_frame_geometry_and_two_action_axes():
    class TwoAxisRunner:
        def infer_tensors(self,image,state,noise):
            self.inputs=(image.copy(),state.copy(),noise.copy())
            return np.tile(np.array([[-2.,2.]],np.float32),(24,1))
        def prepare_tensors(self,mode): return mode
    adapter={'state_mean':[0.]*62,'state_std':[1.]*62,'initial_noise':np.zeros((1,24,2)).tolist(),
             'image_mean':[.5,.5,.5],'image_std':[.5,.5,.5]}
    cfg={'action_axes':['steering','ry'],'weight_branch':'model','image_shape':[240,320,3]}
    p=KartDpPolicy(cfg,adapter,TwoAxisRunner(),model_variant='f32_fast')
    image=np.zeros((4,240,320,3),np.uint8);image[:,:,:,0]=255
    result=p.infer({'observation.images.front':image,'observation.state':np.zeros((4,62))})
    x,_,noise=p.model_runner.inputs
    assert x.shape==(4,3,252,322) and noise.shape==(1,24,2)
    np.testing.assert_allclose(x[0,:,0,0],[1.,-1.,-1.])
    assert result['prediction'].shape==(24,2)
    np.testing.assert_array_equal(result['actions'][0],[-1.,1.])

def test_1004_f32_fast_accepts_gpu_frames_padded_on_device():
    class TwoAxisDeviceRunner:
        def infer_device_pixels(self,images,state,mean,std,noise):
            assert images is frames
            assert all(x.__cuda_array_interface__['shape']==(252,320,3) for x in images)
            assert state.shape==(1,248) and noise.shape==(1,24,2)
            self.mean,self.std=mean,std
            return np.zeros((24,2),np.float32)
        def prepare_tensors(self,mode): return mode
    class Frame:
        def __init__(self):
            self.__cuda_array_interface__={"version":3,"shape":(252,320,3),"typestr":"|u1",
                "data":(123,True),"strides":None,"stream":None}
    frames=[Frame() for _ in range(4)]
    adapter={'state_mean':[0.]*62,'state_std':[1.]*62,'initial_noise':np.zeros((1,24,2)).tolist(),
             'image_mean':[.485,.456,.406],'image_std':[.229,.224,.225]}
    cfg={'action_axes':['steering','ry'],'weight_branch':'model','image_shape':[240,320,3]}
    runner=TwoAxisDeviceRunner();p=KartDpPolicy(cfg,adapter,runner,model_variant='f32_fast')
    result=p.infer({'observation.images.front':frames,'observation.state':np.zeros((4,62),np.float32)})
    assert result['actions'].shape==(24,2)
    np.testing.assert_allclose(runner.mean,[.485,.456,.406])

def test_device_observation_retains_exporters_and_uses_normal_state_contract():
    p=policy();p._gpu_rgb=True
    class Frame:
        __cuda_array_interface__={"version":3,"shape":(224,288,3),"typestr":"|u1",
                                  "data":(123,True),"strides":None,"stream":None}
        def __array__(self,*args,**kwargs):raise AssertionError("device images must not be coerced to NumPy")
    frames=[Frame() for _ in range(4)]
    class DeviceRunner(Runner):
        def infer_device_pixels(self,images,state,mean,std,noise):
            assert images is frames
            np.testing.assert_array_equal(state,np.ones((1,248)))
            return np.zeros((24,3),np.float32)
    p.model_runner=DeviceRunner()
    ob={"observation.images.front":frames,"observation.state":np.full((4,62),3,np.float32)}
    assert p.infer(ob)["actions"].shape==(24,3)
    p._gpu_rgb=False
    with pytest.raises(ValueError,match="native f32_fast"):p.infer(ob)
