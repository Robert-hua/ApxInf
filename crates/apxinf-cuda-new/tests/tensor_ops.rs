#![cfg(feature = "tensor-ops")]
use apxinf_cuda_next::tensor_ops::{Context, Operation};

fn run(ops: &[Operation]) {
    for op in ops { op.run().unwrap(); }
}
fn close(a: &[f32], b: &[f32], tolerance: f32) {
    assert_eq!(a.len(), b.len());
    for (a,b) in a.iter().zip(b) { assert!((a-b).abs() <= tolerance, "{a} != {b}"); }
}

#[test]
fn graph_owns_storage_and_accepts_new_inputs() {
    let ctx = Context::with_bf16(0).unwrap();
    let x = ctx.zeros(&[1,2]).unwrap();
    let w = ctx.tensor(&[2,2], &[1.,2.,3.,4.]).unwrap();
    let (y, op) = ctx.linear(&x,&w,None).unwrap();
    assert!(w.write(&[0.;4]).is_err());
    assert!(w.view(0,&[2]).unwrap().write(&[0.;2]).is_err());
    x.write(&[1.,2.]).unwrap(); op.run().unwrap(); close(&y.read().unwrap(), &[5.,11.],0.);
    let graph = ctx.capture(&[op]).unwrap();
    let previous = y.read().unwrap();
    x.write(&[2.,1.]).unwrap(); graph.replay().unwrap(); close(&y.read().unwrap(), &[4.,10.],0.);
    close(&previous,&[5.,11.],0.);
    drop(w); drop(ctx); graph.replay().unwrap(); close(&y.read().unwrap(), &[4.,10.],0.);
}

#[test]
fn convolution_layout_and_transpose_match_direct_definition() {
    for bf16 in [false,true] {
        let ctx = if bf16 {Context::with_bf16(0)} else {Context::new(0)}.unwrap();
        let xv=(0..2*2*3*4).map(|i|((i%7) as f32-3.)*0.25).collect::<Vec<_>>();
        let wv=(0..3*2*2*2).map(|i|((i%5) as f32-2.)*0.25).collect::<Vec<_>>();
        let x=ctx.tensor(&[2,2,3,4],&xv).unwrap();
        let w=ctx.tensor(&[3,2,2,2],&wv).unwrap();
        let (y,op)=ctx.conv2d(&x,&w,None,[1,1],[0,0],false).unwrap(); op.run().unwrap();
        let mut expected=vec![0.;2*3*2*3];
        for n in 0..2 {for co in 0..3 {for h in 0..2 {for t in 0..3 {for ci in 0..2 {for kh in 0..2 {for kw in 0..2 {
            expected[((n*3+co)*2+h)*3+t] += xv[((n*2+ci)*3+h+kh)*4+t+kw]*wv[((co*2+ci)*2+kh)*2+kw];
        }}}}}}}
        close(&y.read().unwrap(),&expected,1e-6);
        let (z,op)=ctx.conv2d(&y,&w,None,[1,1],[0,0],true).unwrap();op.run().unwrap();
        let mut reverse=vec![0.;xv.len()];
        for n in 0..2 {for co in 0..3 {for h in 0..2 {for t in 0..3 {for ci in 0..2 {for kh in 0..2 {for kw in 0..2 {
            reverse[((n*2+ci)*3+h+kh)*4+t+kw] += expected[((n*3+co)*2+h)*3+t]*wv[((co*2+ci)*2+kh)*2+kw];
        }}}}}}}
        // Small binary fractions remain exactly representable in BF16 here.
        close(&z.read().unwrap(),&reverse,if bf16{0.015625}else{1e-6});
    }
}

#[test]
fn rejects_cross_context_and_partial_aliasing() {
    let ctx=Context::new(0).unwrap();let other=Context::new(0).unwrap();
    let x=ctx.zeros(&[8]).unwrap();let foreign=other.zeros(&[8]).unwrap();
    assert!(ctx.add(&x,&foreign).is_err());
    let (_,op)=other.add(&foreign,&foreign).unwrap();assert!(ctx.capture(&[op]).is_err());
    assert!(ctx.copy_into(&x.view(0,&[4]).unwrap(),&x.view(1,&[4]).unwrap()).is_err());
    let constant=ctx.tensor(&[8],&[0.;8]).unwrap();assert!(ctx.copy_into(&x,&constant).is_err());
    assert!(ctx.linear(&ctx.zeros(&[1,2]).unwrap(),&ctx.zeros(&[2,2]).unwrap(),None).is_err());
}

#[test]
fn group_norm_softmax_and_in_place_sampler_are_finite_and_correct() {
    let ctx=Context::new(0).unwrap();let x=ctx.tensor(&[1,2,2],&[1.,2.,3.,4.]).unwrap();
    let w=ctx.tensor(&[2],&[1.,1.]).unwrap();let b=ctx.tensor(&[2],&[0.,0.]).unwrap();
    let (n,nop)=ctx.norm(&x,&w,&b,4,2,1e-5).unwrap();
    let (s,sop)=ctx.softmax(&x,1.).unwrap();run(&[nop,sop]);
    let inv=1./(1.25f32+1e-5).sqrt();close(&n.read().unwrap(),&[-1.5*inv,-0.5*inv,0.5*inv,1.5*inv],1e-6);
    let prob=1./(1.+1f32.exp());close(&s.read().unwrap(),&[prob,1.-prob,prob,1.-prob],1e-6);
    let dst=ctx.zeros(&[1,2,2]).unwrap();ctx.copy_into(&x,&dst).unwrap().run().unwrap();
    ctx.axpby_into(&dst,&x,None,&dst,[0.5,0.5,0.],2.5).unwrap().run().unwrap();
    close(&dst.read().unwrap(),&[1.,2.,2.5,2.5],0.);
}

#[test]
fn rgb_preprocessing_matches_host_definition_and_updates_graph_input() {
    let ctx=Context::new(0).unwrap();let output=ctx.zeros(&[1,3,2,2]).unwrap();
    let processor=ctx.rgb_processor(&output).unwrap();
    let mean=[0.485,0.456,0.406];let std=[0.229,0.224,0.225];
    let bytes=[0,12,255,128,64,32,255,255,255,1,2,3];
    processor.write(&bytes,mean,std).unwrap();
    let expected=(0..12).map(|i|((bytes[(i%4)*3+i/4] as f32)/255.-mean[i/4])/std[i/4]).collect::<Vec<_>>();
    close(&output.read().unwrap(),&expected,5e-7);
    assert!(processor.write(&bytes,[0.;3],[0.;3]).is_err());
    processor.write(&[0;12],mean,std).unwrap();
    close(&output.read().unwrap(),&(0..12).map(|i|-mean[i/4]/std[i/4]).collect::<Vec<_>>(),5e-7);
}

#[test]
fn tuning_is_explicit_and_cannot_invalidate_graph_storage() {
    let ctx=Context::with_bf16(0).unwrap();
    let x=ctx.tensor(&[1,3,8,8],&vec![0.5;192]).unwrap();
    let w=ctx.tensor(&[8,3,3,3],&vec![0.25;216]).unwrap();
    let(y,op)=ctx.conv2d(&x,&w,None,[1,1],[1,1],false).unwrap();
    op.run().unwrap();let before=y.read().unwrap();op.tune().unwrap();op.run().unwrap();
    close(&y.read().unwrap(),&before,1e-5);
    let graph=ctx.capture(&[op.clone()]).unwrap();op.tune().unwrap();graph.replay().unwrap();
    close(&y.read().unwrap(),&before,1e-5);
    let(_,untuned)=ctx.conv2d(&x,&w,None,[1,1],[1,1],false).unwrap();
    untuned.run().unwrap();let _graph=ctx.capture(&[untuned.clone()]).unwrap();assert!(untuned.tune().is_err());
}

#[test]
fn frozen_batch_norm_preserves_fp32_output_in_bf16_context() {
    let ctx = Context::with_bf16(0).unwrap();
    let x = ctx.tensor(&[1, 2, 1, 2], &[0.5, 1.5, -0.5, 2.0]).unwrap();
    let parameters = ctx.tensor(&[4, 2], &[1.3, 0.7, 0.123, -0.321, 0.2, -0.1, 0.9, 1.1]).unwrap();
    let (output, op) = ctx.frozen_batch_norm(&x, &parameters, 1e-5).unwrap();
    op.run().unwrap();
    let expected = [0.5f32, 1.5, -0.5, 2.0].iter().enumerate().map(|(i, x)| {
        let c=i/2;let scale=[1.3f32,0.7][c]/([0.9f32,1.1][c]+1e-5).sqrt();
        x*scale+([0.123f32,-0.321][c]-[0.2f32,-0.1][c]*scale)
    }).collect::<Vec<_>>();
    let values = output.read().unwrap();
    close(&values, &expected, 1e-6);
    assert!(values.iter().any(|v| half::bf16::from_f32(*v).to_f32() != *v));
    let graph=ctx.capture(&[op]).unwrap();graph.replay().unwrap();
    close(&output.read().unwrap(), &values, 0.);
    assert!(ctx.frozen_batch_norm(&x, &parameters, f32::NAN).is_err());
}

#[test]
fn group_norm_large_rows_match_independent_f64_statistics() {
    let ctx=Context::new(0).unwrap();
    let values=(0..2048).map(|i| 3.0+((i*17%127) as f32-63.0)*0.013).collect::<Vec<_>>();
    let x=ctx.tensor(&[1,8,256], &values).unwrap();
    let weights=vec![0.7f32;8];let biases=vec![-0.1f32;8];
    let w=ctx.tensor(&[8],&weights).unwrap();let b=ctx.tensor(&[8],&biases).unwrap();
    let (output,op)=ctx.norm(&x,&w,&b,1024,256,1e-5).unwrap();op.run().unwrap();
    let mut expected=Vec::new();
    for group in values.chunks(1024){
        let mean=group.iter().map(|v|*v as f64).sum::<f64>()/1024.;
        let var=group.iter().map(|v|(*v as f64-mean).powi(2)).sum::<f64>()/1024.;
        expected.extend(group.iter().map(|v|(((*v as f64-mean)/(var+1e-5).sqrt())*0.7-0.1)as f32));
    }
    close(&output.read().unwrap(),&expected,3e-6);
    let graph=ctx.capture(&[op]).unwrap();graph.replay().unwrap();close(&output.read().unwrap(),&expected,3e-6);
}

#[test]
fn conv1d_real_shape_stride_bias_and_graph_match_f64_reference() {
    // Run with default cuDNN and APXINF_TENSOR_FP32_CONV1D=im2col independently.
    for (batch,cin,cout,length,stride) in [(2,7,11,13,2),(1,256,256,56,1)] {
        let ctx=Context::new(0).unwrap();let kernel=5;let pad=2;
        let out_length=(length+2*pad-kernel)/stride+1;
        let xv=(0..batch*cin*length).map(|i|((i*17%113) as f32-56.)*0.003).collect::<Vec<_>>();
        let wv=(0..cout*cin*kernel).map(|i|((i*13%97) as f32-48.)*0.002).collect::<Vec<_>>();
        let bv=(0..cout).map(|i|i as f32*0.0001).collect::<Vec<_>>();
        let x=ctx.zeros(&[batch,cin,1,length]).unwrap();x.write(&xv).unwrap();
        let w=ctx.tensor(&[cout,cin,1,kernel],&wv).unwrap();let bias=ctx.tensor(&[cout],&bv).unwrap();
        let (y,op)=ctx.conv2d(&x,&w,Some(&bias),[1,stride],[0,pad],false).unwrap();
        let (gemm,gop)=ctx.conv1d_im2col(&x,&w,Some(&bias),stride,pad).unwrap();
        let mut expected=vec![0.;batch*cout*out_length];
        for n in 0..batch {for co in 0..cout {for t in 0..out_length {
            let mut sum=bv[co] as f64;
            for ci in 0..cin {for k in 0..kernel {
                let pos=(t*stride+k) as isize-pad as isize;
                if pos>=0 && pos<length as isize {sum+=xv[(n*cin+ci)*length+pos as usize] as f64*wv[(co*cin+ci)*kernel+k] as f64;}
            }}
            expected[(n*cout+co)*out_length+t]=sum as f32;
        }}}
        op.run().unwrap();close(&y.read().unwrap(),&expected,2e-5);
        gop.run().unwrap();close(&gemm.read().unwrap(),&expected,2e-5);
        let graph=ctx.capture(&[op,gop]).unwrap();
        x.write(&vec![0.;xv.len()]).unwrap();graph.replay().unwrap();
        close(&y.read().unwrap(),&(0..expected.len()).map(|i|bv[(i/out_length)%cout]).collect::<Vec<_>>(),1e-6);
        close(&gemm.read().unwrap(),&y.read().unwrap(),1e-6);
        x.write(&xv).unwrap();graph.replay().unwrap();close(&y.read().unwrap(),&expected,2e-5);
        close(&gemm.read().unwrap(),&expected,2e-5);
    }
}

#[test]
fn explicit_im2col_rejects_other_precisions() {
    for ctx in [Context::with_tf32(0).unwrap(),Context::with_bf16(0).unwrap()] {
        let x=ctx.zeros(&[1,2,1,4]).unwrap();
        let w=ctx.tensor(&[2,2,1,3],&[1.;12]).unwrap();
        assert!(ctx.conv1d_im2col(&x,&w,None,1,1).is_err());
    }
}

#[test]
fn bf16_conv1d_rounding_stride_bias_and_replay() {
    // Same test covers cuDNN, prepared im2row, and the NCHW candidate in separate processes.
    for (batch,cin,cout,length,stride) in [(2,7,11,13,2),(1,128,128,28,1),(1,256,128,28,1)] {
        let ctx=Context::with_bf16(0).unwrap();let kernel=5;let pad=2;
        let olen=(length+2*pad-kernel)/stride+1;
        let round=|x:f32|half::bf16::from_f32(x).to_f32();
        let xv=(0..batch*cin*length).map(|i|((i*17%113)as f32-56.)/256.).collect::<Vec<_>>();
        let wv=(0..cout*cin*kernel).map(|i|((i*13%97)as f32-48.)/512.).collect::<Vec<_>>();
        let bv=(0..cout).map(|i|i as f32/2048.).collect::<Vec<_>>();
        let x=ctx.zeros(&[batch,cin,1,length]).unwrap();x.write(&xv).unwrap();
        let w=ctx.tensor(&[cout,cin,1,kernel],&wv).unwrap();let bias=ctx.tensor(&[cout],&bv).unwrap();
        let (y,op)=ctx.conv2d(&x,&w,Some(&bias),[1,stride],[0,pad],false).unwrap();
        let mut expected=vec![0.;batch*cout*olen];
        for n in 0..batch {for co in 0..cout {for t in 0..olen {
            let mut sum=0f64;
            for ci in 0..cin {for k in 0..kernel {
                let pos=(t*stride+k)as isize-pad as isize;
                if pos>=0&&pos<length as isize{sum+=round(xv[(n*cin+ci)*length+pos as usize])as f64*round(wv[(co*cin+ci)*kernel+k])as f64;}
            }}
            expected[(n*cout+co)*olen+t]=round(round(sum as f32)+round(bv[co]));
        }}}
        op.run().unwrap();close(&y.read().unwrap(),&expected,0.);
        let graph=ctx.capture(&[op]).unwrap();x.write(&vec![0.;xv.len()]).unwrap();graph.replay().unwrap();
        close(&y.read().unwrap(),&(0..expected.len()).map(|i|round(bv[(i/olen)%cout])).collect::<Vec<_>>(),0.);
    }
}

#[test]
fn fp8_conv1d_quantized_definition_and_dynamic_replay() {
    let ctx=Context::with_fp8_conv1d(0).unwrap();
    // Exact E4M3 values, known absmax=448: dequant scales are exactly one.
    // This tests real native FP8 GEMM, padding, batch, stride, bias and input updates.
    let (batch,cin,cout,length,kernel,stride,pad)=(2,128,128,13,5,2,2);
    let olen=(length+2*pad-kernel)/stride+1;
    let values=[-448f32,-32.,-1.,0.,1.,32.,448.];
    let xv=(0..batch*cin*length).map(|i|values[(i*3)%values.len()]).collect::<Vec<_>>();
    let wv=(0..cout*cin*kernel).map(|i|values[(i*5+1)%values.len()]).collect::<Vec<_>>();
    let bv=vec![0.25;cout];let x=ctx.zeros(&[batch,cin,1,length]).unwrap();x.write(&xv).unwrap();
    let w=ctx.tensor(&[cout,cin,1,kernel],&wv).unwrap();let bias=ctx.tensor(&[cout],&bv).unwrap();
    let(y,op)=ctx.conv2d(&x,&w,Some(&bias),[1,stride],[0,pad],false).unwrap();
    let round=|x:f32|half::bf16::from_f32(x).to_f32();
    let expected_for=|factor:f32|{
        let mut expected=vec![0.;batch*cout*olen];
        for n in 0..batch {for co in 0..cout {for t in 0..olen {
            let mut sum=0f64;
            for ci in 0..cin {for k in 0..kernel {
                let pos=(t*stride+k)as isize-pad as isize;
                if pos>=0&&pos<length as isize{sum+=xv[(n*cin+ci)*length+pos as usize]as f64*factor as f64*wv[(co*cin+ci)*kernel+k]as f64;}
            }}
            expected[(n*cout+co)*olen+t]=round(round(sum as f32)+0.25);
        }}}expected
    };
    op.run().unwrap();close(&y.read().unwrap(),&expected_for(1.),0.);
    let graph=ctx.capture(&[op]).unwrap();
    x.write(&xv.iter().map(|v|v*0.5).collect::<Vec<_>>()).unwrap();graph.replay().unwrap();
    close(&y.read().unwrap(),&expected_for(0.5),0.);
    x.write(&vec![0.;xv.len()]).unwrap();graph.replay().unwrap();close(&y.read().unwrap(),&vec![0.25;batch*cout*olen],0.);
}

#[test]
fn bf16_linear_fused_bias_and_graph_match_independent_definition() {
    // Odd K exercises the DP condition projection; multiple rows exercise ACT.
    for (rows, width, outputs) in [(1,225,1024),(4,128,32),(1,2,1)] {
        let ctx=Context::with_bf16(0).unwrap();
        let round=|v:f32|half::bf16::from_f32(v).to_f32();
        let xv=(0..rows*width).map(|i|((i*13%37)as f32-18.)/64.).collect::<Vec<_>>();
        let wv=(0..outputs*width).map(|i|((i*7%29)as f32-14.)/128.).collect::<Vec<_>>();
        let bv=(0..outputs).map(|i|((i%17)as f32-8.)/512.).collect::<Vec<_>>();
        let x=ctx.zeros(&[rows,width]).unwrap();x.write(&xv).unwrap();
        let w=ctx.tensor(&[outputs,width],&wv).unwrap();let bias=ctx.tensor(&[outputs],&bv).unwrap();
        let(y,op)=ctx.linear(&x,&w,Some(&bias)).unwrap();
        let mut expected=vec![0.;rows*outputs];
        for m in 0..rows {for n in 0..outputs {
            let mut sum=round(bv[n])as f64;
            for k in 0..width {sum+=round(xv[m*width+k])as f64*round(wv[n*width+k])as f64;}
            expected[m*outputs+n]=round(sum as f32);
        }}
        op.run().unwrap();close(&y.read().unwrap(),&expected,0.);
        let graph=ctx.capture(&[op]).unwrap();x.write(&vec![0.;xv.len()]).unwrap();
        drop(w);drop(bias);drop(ctx);graph.replay().unwrap();
        close(&y.read().unwrap(),&(0..rows*outputs).map(|i|round(bv[i%outputs])).collect::<Vec<_>>(),0.);
    }
}

#[test]
fn explicit_bf16_convolution_in_fp8_context_preserves_precision_and_graph_lifetime() {
    use apxinf_cuda_next::tensor_ops::ConvPrecision;
    let values=(0..128*17).map(|i|((i%61)as f32-30.)*0.0137).collect::<Vec<_>>();
    let weights=(0..128*128*3).map(|i|((i%37)as f32-18.)*0.00231).collect::<Vec<_>>();
    let reference=Context::with_bf16(0).unwrap();
    let rx=reference.tensor(&[1,128,1,17],&values).unwrap();
    let rw=reference.tensor(&[128,128,1,3],&weights).unwrap();
    let (ry,rop)=reference.conv2d(&rx,&rw,None,[1,1],[0,1],false).unwrap();rop.run().unwrap();
    let expected=ry.read().unwrap();
    let mixed=Context::with_fp8_conv1d(0).unwrap();
    let x=mixed.zeros(&[1,128,1,17]).unwrap();x.write(&values).unwrap();
    let w=mixed.tensor(&[128,128,1,3],&weights).unwrap();
    let (y,op)=mixed.conv2d_with_precision(&x,&w,None,[1,1],[0,1],false,ConvPrecision::Bf16).unwrap();
    let (q,qop)=mixed.conv2d(&x,&w,None,[1,1],[0,1],false).unwrap();
    op.run().unwrap();qop.run().unwrap();close(&y.read().unwrap(),&expected,0.);
    assert!(q.read().unwrap().iter().zip(&expected).any(|(a,b)|a!=b));
    let graph=mixed.capture(&[op,qop]).unwrap();drop(w);drop(mixed);
    x.write(&vec![0.;values.len()]).unwrap();graph.replay().unwrap();close(&y.read().unwrap(),&vec![0.;expected.len()],0.);
    x.write(&values).unwrap();graph.replay().unwrap();close(&y.read().unwrap(),&expected,0.);
    let f32ctx=Context::new(0).unwrap();
    assert!(f32ctx.conv2d_with_precision(&rx,&rw,None,[1,1],[0,1],false,ConvPrecision::Bf16).is_err());
}

#[test]
fn exact_gelu_matches_erf_golden_and_rebinds_graph_input() {
    use apxinf_cuda_next::tensor_ops::Activation;
    // Independent double precision erf definition, rounded to f32.
    let input=[-8.,-3.,-1.,-0.1,0.,0.1,1.,3.,8.];
    let golden=[-4.8849813e-15,-0.004049694,-0.15865526,-0.046017215,0.,0.053982783,0.8413448,2.9959502,8.];
    let ctx=Context::new(0).unwrap();let x=ctx.zeros(&[9]).unwrap();x.write(&input).unwrap();
    let (y,op)=ctx.activation(&x,Activation::Gelu).unwrap();op.run().unwrap();
    close(&y.read().unwrap(),&golden,3e-7);
    let graph=ctx.capture(&[op]).unwrap();let old=y.read().unwrap();
    x.write(&[0.;9]).unwrap();graph.replay().unwrap();close(&y.read().unwrap(),&[0.;9],0.);
    close(&old,&golden,3e-7);
}

#[test]
fn opt_in_block_reductions_preserve_broadcast_before_reusing_shared_memory() {
    let ctx=Context::new(0).unwrap();
    for width in [337usize,384] {
        let rows=32;
        let values=(0..rows*width).map(|i|((i*13%101) as f32-50.)*0.125+(i/width) as f32).collect::<Vec<_>>();
        let x=ctx.tensor(&[rows,width],&values).unwrap();
        let w=ctx.tensor(&[width],&vec![1.;width]).unwrap();
        let b=ctx.tensor(&[width],&vec![0.;width]).unwrap();
        let (norm,nop)=ctx.layer_norm_block(&x,&w,&b,1e-6).unwrap();
        let (prob,sop)=ctx.softmax_block(&x,1.).unwrap();
        let mut norms=Vec::new();let mut probs=Vec::new();
        for row in values.chunks_exact(width) {
            let mean=row.iter().map(|&v|v as f64).sum::<f64>()/width as f64;
            let var=row.iter().map(|&v|(v as f64-mean).powi(2)).sum::<f64>()/width as f64;
            let max=row.iter().copied().fold(f32::NEG_INFINITY,f32::max) as f64;
            let sum=row.iter().map(|&v|(v as f64-max).exp()).sum::<f64>();
            norms.extend(row.iter().map(|&v|((v as f64-mean)/(var+1e-6).sqrt()) as f32));
            probs.extend(row.iter().map(|&v|((v as f64-max).exp()/sum) as f32));
        }
        let ops=[nop,sop];
        for _ in 0..64 {
            run(&ops);
            close(&norm.read().unwrap(),&norms,2e-6);
            close(&prob.read().unwrap(),&probs,2e-7);
        }
        let graph=ctx.capture(&ops).unwrap();graph.replay().unwrap();
        close(&norm.read().unwrap(),&norms,2e-6);
        close(&prob.read().unwrap(),&probs,2e-7);
    }
}

#[test]
fn block_reduction_contract_rejects_invalid_affine_and_scale() {
    let ctx=Context::new(0).unwrap();
    let x=ctx.tensor(&[2,3],&[1.;6]).unwrap();
    let w=ctx.tensor(&[3],&[1.;3]).unwrap();
    let wrong=ctx.tensor(&[1,3],&[0.;3]).unwrap();
    assert!(ctx.layer_norm_block(&x,&w,&wrong,1e-6).is_err());
    assert!(ctx.layer_norm_block(&x,&w,&w,0.).is_err());
    assert!(ctx.softmax_block(&x,f32::NAN).is_err());
}
