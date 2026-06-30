import os
import torch
import yaml
import pprint
from src.helper import init_model, init_opt
from src.masks.multiblock import MaskCollator as MBMaskCollator
import torch.nn.functional as F
from src.masks.utils import apply_masks
from src.utils.tensors import repeat_interleave_batch

def run_profiling():
    print("Starting profiling setup...")
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load test config
    config_path = 'configs/test_stress.yaml'
    # Fallback to test_vith14 if test_stress doesn't exist
    if not os.path.exists(config_path):
        config_path = 'configs/test_vith14.yaml'
        
    with open(config_path, 'r') as f:
        args = yaml.load(f, Loader=yaml.FullLoader)
    
    # Extract params
    crop_size = args['data']['crop_size']
    patch_size = args['mask']['patch_size']
    batch_size = args['data']['batch_size']
    pred_depth = args['meta']['pred_depth']
    pred_emb_dim = args['meta']['pred_emb_dim']
    model_name = args['meta']['model_name']
    use_bfloat16 = args['meta']['use_bfloat16']
    
    # Overrides for quick profiling
    args['optimization']['epochs'] = 1
    if device.type == 'cpu':
        model_name = 'vit_tiny'
        pred_depth = 2
        pred_emb_dim = 192
        batch_size = 2
        use_bfloat16 = False
    
    print(f"Profiling config: model={model_name}, batch_size={batch_size}, bfloat16={use_bfloat16}")
    
    # Init models
    encoder, predictor = init_model(
        device=device,
        patch_size=patch_size,
        crop_size=crop_size,
        pred_depth=pred_depth,
        pred_emb_dim=pred_emb_dim,
        model_name=model_name
    )
    import copy
    target_encoder = copy.deepcopy(encoder)
    for p in target_encoder.parameters():
        p.requires_grad = False
        
    # Init optimizers
    optimizer, scaler, scheduler, wd_scheduler = init_opt(
        encoder=encoder,
        predictor=predictor,
        wd=float(args['optimization']['weight_decay']),
        final_wd=float(args['optimization']['final_weight_decay']),
        start_lr=args['optimization']['start_lr'],
        ref_lr=args['optimization']['lr'],
        final_lr=args['optimization']['final_lr'],
        iterations_per_epoch=10,
        warmup=args['optimization']['warmup'],
        num_epochs=1,
        ipe_scale=1.0,
        use_bfloat16=use_bfloat16,
        optimized_code=True
    )
    
    if device.type == 'cuda' and hasattr(torch, 'compile'):
        print("Compiling models for profiling...")
        encoder = torch.compile(encoder, dynamic=False)
        predictor = torch.compile(predictor, dynamic=False)
        target_encoder = torch.compile(target_encoder, dynamic=False)
    
    # Collator & dummy batch
    mask_collator = MBMaskCollator(
        input_size=crop_size,
        patch_size=patch_size,
        pred_mask_scale=args['mask']['pred_mask_scale'],
        enc_mask_scale=args['mask']['enc_mask_scale'],
        aspect_ratio=args['mask']['aspect_ratio'],
        nenc=args['mask']['num_enc_masks'],
        npred=args['mask']['num_pred_masks'],
        allow_overlap=args['mask']['allow_overlap'],
        min_keep=args['mask']['min_keep']
    )
    
    # Mock data batch
    dummy_imgs = torch.randn(batch_size, 3, crop_size, crop_size)
    dummy_batch = [(img, 0) for img in dummy_imgs]
    
    # Setup profiler
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == 'cuda':
        activities.append(torch.profiler.ProfilerActivity.CUDA)
        
    os.makedirs('data', exist_ok=True)
    trace_path = 'data/ijepa_profile_trace.json'
    
    print("Warming up...")
    # Warmup step
    for _ in range(10):
        # Generate masks
        udata, masks_enc, masks_pred = mask_collator(dummy_batch)
        imgs = udata[0].to(device)
        m_enc = [u.to(device) for u in masks_enc]
        m_pred = [u.to(device) for u in masks_pred]
        
        # Forward Context
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bfloat16):
            z = encoder(imgs, m_enc)
            z = predictor(z, m_enc, m_pred)
            
            # Forward Target
            with torch.no_grad():
                h = target_encoder(imgs)
                h = F.layer_norm(h, (h.size(-1),))
                h = apply_masks(h, m_pred)
                h = repeat_interleave_batch(h, len(h), repeat=len(m_enc))
                
            loss = F.smooth_l1_loss(z, h)
            
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        optimizer.zero_grad()

    print("Starting profiling run...")
    with torch.profiler.profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:
        for step in range(20):
            print(f"Step {step+1}/20...")
            with torch.profiler.record_function("mask_collation"):
                udata, masks_enc, masks_pred = mask_collator(dummy_batch)
                imgs = udata[0].to(device)
                m_enc = [u.to(device) for u in masks_enc]
                m_pred = [u.to(device) for u in masks_pred]
                
            with torch.profiler.record_function("forward_pass"):
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bfloat16):
                    with torch.profiler.record_function("encoder_context"):
                        z = encoder(imgs, m_enc)
                    with torch.profiler.record_function("predictor_context"):
                        z = predictor(z, m_enc, m_pred)
                    
                    with torch.profiler.record_function("target_encoder"):
                        with torch.no_grad():
                            h = target_encoder(imgs)
                            h = F.layer_norm(h, (h.size(-1),))
                            h = apply_masks(h, m_pred)
                            h = repeat_interleave_batch(h, len(h), repeat=len(m_enc))
                            
                    with torch.profiler.record_function("loss_computation"):
                        loss = F.smooth_l1_loss(z, h)
                        
            with torch.profiler.record_function("backward_pass"):
                if scaler is not None:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                    
            with torch.profiler.record_function("optimizer_step"):
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()
                
            prof.step()

    print(f"Exporting Chrome Tracing JSON to {trace_path}...")
    prof.export_chrome_trace(trace_path)
    
    print("\n--- PROFILING RESULTS SUMMARY (by CPU time) ---")
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=25))
    
    if device.type == 'cuda':
        print("\n--- GPU OPERATORS SUMMARY (by CUDA time) ---")
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=25))

if __name__ == '__main__':
    run_profiling()
