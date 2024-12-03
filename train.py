import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from fake_celeb_dataset import FakeAVceleb
from model import MP_AViT
from backbone.select_backbone import select_backbone
from audio_process import AudioEncoder
from load_audio import wav2filterbanks
from config_deepfake import load_opts
import os
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR
import logging
import torch.nn.functional as F

# Network类定义
class Network(torch.nn.Module):
    def __init__(self, vis_enc, aud_enc, transformer):
        super().__init__()
        self.vis_enc = vis_enc
        self.aud_enc = aud_enc
        self.transformer = transformer

    def forward(self, video, audio, phase=0, train=True):
        if train:
            if phase == 0:  # 不同视频的负样本
                vid_emb = self.vis_enc(video)
                batch_size, c, t, h, w = vid_emb.shape
                vid_emb = vid_emb[:, None]
                vid_emb = vid_emb.expand(-1, batch_size, -1, -1, -1, -1)
                vid_emb = vid_emb.reshape(-1, c, t, h, w)

                aud_emb = self.aud_enc(audio)
                batch_size, c_aud, t_aud = aud_emb.shape
                aud_emb = aud_emb[None, :]
                aud_emb = aud_emb.expand(batch_size, -1, -1, -1).reshape(-1, c_aud, t_aud)

                cls_emb = self.transformer(vid_emb, aud_emb)

            elif phase == 1:  # 同一视频内的负样本
                vid_emb = self.vis_enc(video)
                batch_size, c, t, h, w = vid_emb.shape
                vid_emb = vid_emb[:, None]
                vid_emb = vid_emb.expand(-1, opts.number_sample, -1, -1, -1, -1)
                vid_emb = vid_emb.reshape(-1, c, t, h, w)

                aud_emb = self.aud_enc(audio)
                batch_size, c_aud, t_aud = aud_emb.shape
                aud_emb_new = torch.zeros_like(aud_emb)
                aud_emb_new = aud_emb_new[None, :]
                aud_emb_new = aud_emb_new.expand(opts.number_sample, -1, -1, -1).reshape(-1, c_aud, t_aud)

                num_sample = opts.number_sample
                bs2 = batch_size // num_sample
                for k in range(bs2):
                    aud_emb_new[k*num_sample*num_sample:(k+1)*num_sample*num_sample] = (
                        aud_emb[k*num_sample:(k+1)*num_sample][None, :]
                    ).expand(opts.number_sample, -1, -1, -1).reshape(-1, c_aud, t_aud)

                aud_emb = aud_emb_new
                cls_emb = self.transformer(vid_emb, aud_emb)
        else:
            vid_emb = self.vis_enc(video)
            aud_emb = self.aud_enc(audio)
            cls_emb = self.transformer(vid_emb, aud_emb)
        return cls_emb

def setup_logger(save_dir):
    logger = logging.getLogger('train')
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    fh = logging.FileHandler(os.path.join(save_dir, 'train.log'))
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger

def train():
    opts = load_opts()  # 加载配置
    device = torch.device(opts.device if torch.cuda.is_available() else "cpu")  # 自动选择CPU或GPU
    logger = setup_logger(opts.output_dir)
    writer = SummaryWriter(os.path.join(opts.output_dir, 'logs'))
    os.makedirs(opts.output_dir, exist_ok=True)

    vis_enc, _ = select_backbone(network='r18')
    aud_enc = AudioEncoder()
    transformer = MP_AViT(
        image_size=14,
        patch_size=0,
        num_classes=1,
        dim=512,
        depth=3,
        heads=4,
        mlp_dim=512,
        dim_head=128,
        dropout=0.1,
        emb_dropout=0.1,
        max_visual_len=14,  # 调整为实际视频帧数
        max_audio_len=4
    )

    model = Network(vis_enc, aud_enc, transformer).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=opts.lr)

    # Phase 1
    logger.info("Starting Phase 1 Training...")
    train_dataset = FakeAVceleb(
        video_path=opts.train_list,
        resize=opts.resize,
        fps=opts.fps,
        sample_rate=opts.sample_rate,
        vid_len=5,  # 与 Transformer 的 max_visual_len 一致
        phase=0,
        train=True
    )

    train_loader = DataLoader(train_dataset, batch_size=16, num_workers=opts.n_workers, shuffle=True)
    scheduler = CosineAnnealingLR(optimizer, T_max=opts.epochs_0)

    for epoch in range(opts.epochs_0):
        model.train()
        total_loss = 0
        for batch_idx, data in enumerate(tqdm(train_loader)):
            video = data['video'].to(device)
            audio = data['audio'].to(device)
            print(f"Video shape: {video.shape}, Audio shape: {audio.shape}")
            audio, _, _, _ = wav2filterbanks(audio, device=device)
            audio = audio.permute([0, 2, 1])[:, None]

            optimizer.zero_grad()
            scores = model(video, audio, phase=0, train=True)
            loss = -torch.mean(torch.log(scores.diagonal()))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            if batch_idx % opts.log_interval == 0:
                logger.info(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item():.4f}')
                writer.add_scalar('Loss/phase1', loss.item(), epoch * len(train_loader) + batch_idx)

        scheduler.step()
        logger.info(f'Epoch {epoch}, Average Loss: {total_loss / len(train_loader):.4f}')
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), os.path.join(opts.output_dir, f'phase1_epoch{epoch}.pth'))

    # 保存最终模型
    torch.save(model.state_dict(), os.path.join(opts.output_dir, 'final_model.pth'))
    writer.close()
    logger.info("Training completed!")

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--train_list', type=str, required=True, help='训练数据列表文件路径')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--output_dir', type=str, default='./checkpoints')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--epochs_0', type=int, default=50)
    parser.add_argument('--epochs_1', type=int, default=90)
    parser.add_argument('--number_sample', type=int, default=4)
    parser.add_argument('--resize', type=int, default=224)
    parser.add_argument('--fps', type=int, default=25)
    parser.add_argument('--sample_rate', type=int, default=16000)
    parser.add_argument('--vid_len', type=int, default=5)
    parser.add_argument('--n_workers', type=int, default=4)
    parser.add_argument('--log_interval', type=int, default=100)
    parser.add_argument('--bs2', type=int, default=40)
    opts = parser.parse_args()

    train()
