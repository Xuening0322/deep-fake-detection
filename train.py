import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam
import torchaudio
import torchaudio.transforms as T
from tqdm import tqdm
import numpy as np
import os
import argparse
from loguru import logger
import traceback

from fake_celeb_dataset_detect import FakeAVceleb
from model import MP_AViT, MP_av_feature_AViT
from backbone.select_backbone import select_backbone
from audio_process import AudioEncoder
from transformer_component import transformer_decoder
from load_audio import wav2filterbanks

def collate_fn(batch):
    """
    Custom collate function that handles variable length sequences by using fixed-length windows
    """
    # Extract fixed-length segments from each video/audio pair
    videos = []
    audios = []
    shifts = []

    for sample in batch:
        video = torch.from_numpy(sample['video']).float()  # [C, T, H, W]
        audio = torch.from_numpy(sample['audio']).float()  # [T*aud_fact]
        shift = torch.from_numpy(sample['shift']).long()

        # We want 5-frame segments
        num_frames = video.shape[1]
        if num_frames >= 5:
            # Randomly select starting point
            max_start = num_frames - 5
            start_idx = np.random.randint(0, max_start + 1)
            
            # Extract video segment
            video_segment = video[:, start_idx:start_idx+5, :, :]
            
            # Extract corresponding audio segment
            aud_fact = 640
            audio_start = start_idx * aud_fact
            audio_end = (start_idx + 5) * aud_fact
            audio_segment = audio[audio_start:audio_end]

            videos.append(video_segment)
            audios.append(audio_segment)
            shifts.append(shift)

    if not videos:  # Skip batch if no valid segments found
        raise RuntimeError("No valid segments found in batch")

    # Stack all segments
    videos = torch.stack(videos)
    audios = torch.stack(audios)
    shifts = torch.stack(shifts)

    return {
        'video': videos,
        'audio': audios,
        'shift': shifts
    }

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_list', type=str, required=True)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--output_dir', type=str, default='./checkpoints')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--epochs_0', type=int, default=50)
    parser.add_argument('--epochs_1', type=int, default=90)
    parser.add_argument('--number_sample', type=int, default=4)
    parser.add_argument('--bs2', type=int, default=40)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=2)  # Reduced from 4 to 2
    parser.add_argument('--save_frequency', type=int, default=5)
    parser.add_argument('--resize', type=int, default=224)
    parser.add_argument('--fps', type=int, default=25)
    parser.add_argument('--sample_rate', type=int, default=16000)
    parser.add_argument('--vid_len', type=int, default=5)
    return parser.parse_args()

class network(nn.Module):
    def __init__(self, vis_enc, aud_enc, transformer):
        super().__init__()
        self.vis_enc = vis_enc
        self.aud_enc = aud_enc
        self.transformer = transformer

    def forward(self, video, audio, phase=0, train=True):
        if train:
            if phase == 0:
                # Get video embeddings
                vid_emb = self.vis_enc(video)  # [B, C, T, H, W]
                batch_size = vid_emb.shape[0]
                
                # Get audio embeddings
                aud_emb = self.aud_enc(audio)  # [B, C_aud, T_aud]
                
                # Use clone() to avoid in-place operations
                vid_emb = vid_emb.clone()
                aud_emb = aud_emb.clone()
                
                cls_emb = self.transformer(vid_emb, aud_emb)
                
            elif phase == 1:
                vid_emb = self.vis_enc(video)
                batch_size = vid_emb.shape[0]
                num_sample = args.number_sample
                
                # Use clone() to avoid in-place operations
                vid_emb = vid_emb.clone()
                
                # Expand video embeddings without in-place operations
                vid_emb = torch.repeat_interleave(vid_emb.unsqueeze(1), 
                                                repeats=num_sample, 
                                                dim=1)
                vid_emb = vid_emb.flatten(0, 1)  # [B*N, C, T, H, W]
                
                # Get audio embeddings
                aud_emb = self.aud_enc(audio)  # [B, C_aud, T_aud]
                aud_emb = aud_emb.clone()
                
                # Expand audio embeddings without in-place operations
                aud_emb = torch.repeat_interleave(aud_emb.unsqueeze(1), 
                                                repeats=num_sample, 
                                                dim=1)
                aud_emb = aud_emb.flatten(0, 1)  # [B*N, C_aud, T_aud]
                
                cls_emb = self.transformer(vid_emb, aud_emb)
        else:
            vid_emb = self.vis_enc(video)
            aud_emb = self.aud_enc(audio)
            cls_emb = self.transformer(vid_emb, aud_emb)
        
        return cls_emb

def process_audio_batch(audio_batch, device):
    """
    Process a batch of audio using torchaudio's mel spectrogram.
    """
    mel_transform = T.MelSpectrogram(
        sample_rate=16000,
        n_fft=1024,
        win_length=1024,
        hop_length=160,
        n_mels=80,
        center=False,
    ).to(device)
    
    # Add a batch dimension if not present
    if audio_batch.dim() == 1:
        audio_batch = audio_batch.unsqueeze(0)
    
    # Process the batch
    mel_specs = []
    for audio in audio_batch:
        # Normalize audio
        audio = audio / (torch.max(torch.abs(audio)) + 1e-8)
        
        # Compute mel spectrogram
        mel_spec = mel_transform(audio)
        
        # Convert to log scale
        mel_spec = torch.log(mel_spec + 1e-8)
        
        mel_specs.append(mel_spec)
    
    # Stack all spectrograms
    mel_specs = torch.stack(mel_specs)
    
    return mel_specs.unsqueeze(1)  # [B, 1, n_mels, T]

def train_phase_0(model, train_loader, optimizer, device, epoch):
    model.train()
    total_loss = 0
    criterion = nn.CrossEntropyLoss()
    valid_batches = 0

    with tqdm(total=len(train_loader)) as pbar:
        for batch_idx, data in enumerate(train_loader):
            try:
                video = data['video'].to(device)
                audio = data['audio'].to(device)  
                shifts = data['shift'].to(device)
                
                # 为每个可能的偏移创建输入
                B = video.shape[0]
                video = video[:, None].repeat(1, 31, 1, 1, 1, 1)  # [B, 31, C, T, H, W] 
                video = video.reshape(-1, *video.shape[2:])  # [B*31, C, T, H, W]

                audio_list = []
                for j in range(B):
                    for i in range(31):  
                        audio_list.append(audio[j:j+1])
                audio = torch.cat(audio_list, dim=0)  # [B*31]
                
                # 音频处理
                audio_processed = process_audio_batch(audio, device)
                
                # 获取异常分数
                scores = model(video, audio_processed, phase=0)  # [B*31, 1]
                scores = scores.reshape(B, 31)  # [B, 31]
                
                # 计算损失 - 用真实的偏移作为目标
                remapped_shifts = shifts + 15
                loss = criterion(scores, remapped_shifts.squeeze())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                valid_batches += 1
                pbar.set_postfix({'loss': f'{total_loss/valid_batches:.4f}'})
                pbar.update(1)

            except Exception as e:
                logger.error(f"Error in batch {batch_idx}: {str(e)}")
                continue

    return total_loss / valid_batches if valid_batches > 0 else float('inf')

def train_phase_1(model, train_loader, optimizer, device, epoch):
    model.train()
    total_loss = 0
    criterion = nn.CrossEntropyLoss()
    valid_batches = 0

    with tqdm(total=len(train_loader)) as pbar:
        for batch_idx, data in enumerate(train_loader):
            try:
                video = data['video'].to(device)
                audio = data['audio'].to(device)  
                shifts = data['shift'].to(device)

                # Debug: Print initial shapes
                logger.debug(f"Initial shapes - video: {video.shape}, audio: {audio.shape}, shifts: {shifts.shape}")

                # 为每个可能的偏移创建输入 
                B = video.shape[0]
                
                # Expand video
                video = video[:, None].repeat(1, 31, 1, 1, 1, 1)  
                video = video.reshape(-1, *video.shape[2:])

                # Debug: Print expanded video shape
                logger.debug(f"Expanded video shape: {video.shape}")

                # Expand audio
                audio_list = [] 
                for j in range(B):
                    for i in range(31):
                        audio_list.append(audio[j:j+1])
                audio = torch.cat(audio_list, dim=0)

                # Debug: Print expanded audio shape
                logger.debug(f"Expanded audio shape: {audio.shape}")

                # 处理音频
                audio_processed = process_audio_batch(audio, device)
                
                # Debug: Print processed audio shape
                logger.debug(f"Processed audio shape: {audio_processed.shape}")
                
                # 获取predictions
                scores = model(video, audio_processed, phase=1)
                
                # Debug: Print scores shape
                logger.debug(f"Raw scores shape: {scores.shape}")

                # 计算loss
                remapped_shifts = shifts + 15
                scores = scores.reshape(B, -1)  # 动态推断第二维度的大小
                
                # Debug: Print final shapes
                logger.debug(f"Final shapes - scores: {scores.shape}, targets: {remapped_shifts.shape}")
                
                loss = criterion(scores, remapped_shifts.squeeze())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                valid_batches += 1
                pbar.set_postfix({'loss': f'{total_loss/valid_batches:.4f}'})
                pbar.update(1)

            except Exception as e:
                logger.error(f"Error in batch {batch_idx}: {str(e)}")
                logger.error(traceback.format_exc())
                continue

    return total_loss / valid_batches if valid_batches > 0 else float('inf')
  
def main():
    # Enable anomaly detection for debugging
    torch.autograd.set_detect_anomaly(True)
    
    global args
    args = parse_args()
    
    # Create output directory and setup logging
    os.makedirs(args.output_dir, exist_ok=True)
    logger.add(os.path.join(args.output_dir, "training.log"))
    
    device = torch.device(args.device)
    logger.info(f"Using device: {device}")
    
    # Load training data
    with open(args.train_list, 'r') as f:
        train_videos = [line.strip() for line in f.readlines()]
    
    logger.info(f"Loaded {len(train_videos)} training videos")
    
    # Initialize models
    vis_enc, _ = select_backbone(network='r18')
    aud_enc = AudioEncoder()
    
    transformer = MP_AViT(image_size=14, patch_size=0, num_classes=1, dim=512, depth=3, heads=4, mlp_dim=512,  dim_head=128, dropout=0.1, emb_dropout=0.1, max_visual_len=5, max_audio_len=4)
    sync_model = network(vis_enc=vis_enc, aud_enc=aud_enc, transformer=transformer)
    sync_model = sync_model.to(device)
    
    optimizer = Adam(sync_model.parameters(), lr=args.lr)
    
    # Phase 0 Training
    logger.info("Starting Phase 0 Training...")
    
    train_dataset_phase0 = FakeAVceleb(
        train_videos,
        args.resize,
        args.fps,
        args.sample_rate,
        vid_len=5,
        phase=0,
        train=True,
        number_sample=1,
        lrs2=True,
        need_shift=True,
        random_shift=True
    )
    
    train_loader_phase0 = DataLoader(
        train_dataset_phase0,
        batch_size=args.batch_size,
        num_workers=0,
        shuffle=True,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    try:
        # Phase 0 Training Loop
        for epoch in range(args.epochs_0):
            loss = train_phase_0(
                sync_model,
                train_loader_phase0,
                optimizer,
                device,
                epoch
            )
            
            logger.info(f'Phase 0 - Epoch {epoch}: Loss = {loss:.4f}')
            
            if (epoch + 1) % args.save_frequency == 0:
                save_path = os.path.join(args.output_dir, f'sync_model_phase0_epoch_{epoch+1}.pth')
                torch.save(sync_model.state_dict(), save_path)
                logger.info(f'Saved checkpoint to {save_path}')
        
        # Save final Phase 0 model
        save_path = os.path.join(args.output_dir, 'sync_model_phase0_final.pth')
        torch.save(sync_model.state_dict(), save_path)
        logger.info("Phase 0 Training completed!")
        
        # Phase 1 Training
        logger.info("Starting Phase 1 Training...")
        
        train_dataset_phase1 = FakeAVceleb(
            train_videos,
            args.resize,
            args.fps,
            args.sample_rate,
            vid_len=5,  # Keep consistent with Phase 0
            phase=1,
            train=True,
            number_sample=args.number_sample,
            lrs2=True,
            need_shift=True,
            random_shift=True
        )
        
        train_loader_phase1 = DataLoader(
            train_dataset_phase1,
            batch_size=args.bs2,
            num_workers=0,
            shuffle=True,
            collate_fn=collate_fn,
            pin_memory=True
        )
        
        # Phase 1 Training Loop
        for epoch in range(args.epochs_1):
            loss = train_phase_1(
                sync_model,
                train_loader_phase1,
                optimizer,
                device,
                epoch
            )
            
            logger.info(f'Phase 1 - Epoch {epoch}: Loss = {loss:.4f}')
            
            if (epoch + 1) % args.save_frequency == 0:
                save_path = os.path.join(args.output_dir, f'sync_model_phase1_epoch_{epoch+1}.pth')
                torch.save(sync_model.state_dict(), save_path)
                logger.info(f'Saved checkpoint to {save_path}')
        
        # Save final model
        save_path = os.path.join(args.output_dir, 'sync_model_final.pth')
        torch.save(sync_model.state_dict(), save_path)
        logger.info("Phase 1 Training completed!")
                
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())

    logger.info("All training completed!")

if __name__ == '__main__':
    main()
