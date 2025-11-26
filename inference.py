# inference.py

import os
import numpy as np
import torch
import torch.nn as nn
from transformers import VideoMAEImageProcessor, VideoMAEModel
import decord
import librosa
from moviepy import VideoFileClip
from scipy.signal import find_peaks
import joblib

# 1. 모델 구조 정의 (Cell 3 내용)
class MLP(nn.Module):
    def __init__(self, d, h=256, p=0.5, n=2):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, h), nn.ReLU(), nn.Dropout(p), nn.Linear(h, n))

    def forward(self, x):
        return self.net(x)

class HighlightDetector:
    def __init__(self, model_path, scaler_path, device='cuda'):
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.num_frames = 16
        
        # 모델 및 스케일러 로드
        print("⏳ 모델 로딩 중...")
        state = torch.load(model_path, map_location=self.device)
        d_in = int(state.get("feat_dim", 848))
        
        self.model = MLP(d=d_in).to(self.device)
        self.model.load_state_dict(state["state_dict"])
        self.model.eval()
        
        self.scaler = joblib.load(scaler_path)
        
        # VideoMAE 로드
        self.processor = VideoMAEImageProcessor.from_pretrained("MCG-NJU/videomae-base")
        self.backbone = VideoMAEModel.from_pretrained("MCG-NJU/videomae-base").to(self.device).eval()
        
        # FP16 설정 (GPU 사용 시)
        if self.device == 'cuda':
            self.backbone.half()
        
        decord.bridge.set_bridge('native')
        print("✅ 모델 로딩 완료")

    def _extract_video_feat(self, vr, s, e):
        # Cell 4의 extract_video_feat_vr 함수 로직
        fps = float(vr.get_avg_fps())
        sf, ef = int(s * fps), max(int(e * fps) - 1, int(s * fps) + 1)
        max_frame_idx = len(vr) - 1
        sf = min(sf, max_frame_idx); ef = min(ef, max_frame_idx)
        
        if sf >= ef: return np.zeros(768, dtype=np.float32)
        
        idx = np.linspace(sf, ef, self.num_frames).astype(int)
        idx = np.clip(idx, 0, max_frame_idx)
        frames = [vr[i].asnumpy() for i in idx]
        
        inputs = self.processor(frames, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        if self.device == 'cuda':
            inputs = {k: v.half() for k, v in inputs.items()}
            
        with torch.no_grad():
            out = self.backbone(**inputs).last_hidden_state.mean(dim=1)
            
        return out.squeeze(0).float().cpu().numpy()

    def _extract_audio_feat(self, y_seg, sr=16000):
        # Cell 4의 extract_audio_feat 함수 로직
        if len(y_seg) < int(0.05 * sr): return np.zeros(80, dtype=np.float32) # 64+16
        
        S = librosa.feature.melspectrogram(y=y_seg, sr=sr, n_mels=64, n_fft=2048, hop_length=int(sr*0.01), win_length=int(sr*0.025))
        L = librosa.power_to_db(S + 1e-9)
        mel_vec = L.mean(axis=1)
        mfcc = librosa.feature.mfcc(S=librosa.power_to_db(S + 1e-9), n_mfcc=16)
        mfcc_vec = mfcc.mean(axis=1)
        
        return np.concatenate([mel_vec, mfcc_vec], axis=0).astype(np.float32)

    def predict(self, video_path):
        # Cell 4의 메인 루프 로직 통합
        try:
            vr = decord.VideoReader(video_path)
            dur = len(vr) / float(vr.get_avg_fps())
            
            # 오디오 로드 (ffmpeg 의존성 주의)
            try:
                y_all, sr = librosa.load(video_path, sr=16000, mono=True)
            except:
                # librosa 실패 시 moviepy 사용 (임시 파일 생성)
                tmp_wav = f"temp_{os.path.basename(video_path)}.wav"
                clip = VideoFileClip(video_path)
                clip.audio.write_audiofile(tmp_wav, fps=16000, verbose=False, logger=None)
                y_all, sr = librosa.load(tmp_wav, sr=16000, mono=True)
                os.remove(tmp_wav)

            # 슬라이딩 윈도우
            win, stride = 5.0, 2.5
            cands = []
            
            starts = np.arange(0.0, max(0.0, dur - win + stride/2), stride)
            
            for t in starts:
                t_end = min(t + win, dur)
                if t_end - t < 0.1: continue
                
                v_feat = self._extract_video_feat(vr, t, t_end)
                
                a_start_idx = int(t * sr)
                a_end_idx = max(a_start_idx + 1, int(t_end * sr))
                y_seg = y_all[a_start_idx:a_end_idx]
                a_feat = self._extract_audio_feat(y_seg, sr)
                
                fused = np.concatenate([v_feat, a_feat], axis=0).astype("float32")
                
                # 정규화 및 추론
                xf = self.scaler.transform(fused[None, :])
                x_tensor = torch.from_numpy(xf).to(self.device).float()
                
                with torch.no_grad():
                    score = self.model(x_tensor).softmax(dim=1)[0, 1].item()
                
                cands.append({"start": float(t), "end": float(t_end), "score": float(score)})
                
            # 후처리 (Peak Detection)
            return self._post_process(cands, dur)
            
        except Exception as e:
            print(f"Error processing {video_path}: {e}")
            return []

    def _post_process(self, cands, dur):
        # Cell 4의 후처리 로직 (Peak Detection + Top K)
        if not cands: return []
        
        scores = np.array([c["score"] for c in cands])
        times = np.array([c["start"] + (c["end"] - c["start"])/2 for c in cands])
        
        peaks, _ = find_peaks(scores, height=0.3, distance=max(1, int(15.0/2.5)))
        
        final_highlights = []
        if len(peaks) > 0:
            peak_times = times[peaks]
            peak_scores = scores[peaks]
            
            # Top 5 선택
            top_k_idx = np.argsort(peak_scores)[::-1][:5]
            selected_times = np.sort(peak_times[top_k_idx])
            
            for t in selected_times:
                s = max(0.0, t - 10.0) # 20초 길이 (중앙 기준 +- 10초)
                e = min(dur, s + 20.0)
                s = max(0.0, e - 20.0)
                final_highlights.append({"start": round(s, 2), "end": round(e, 2)})
                
        return final_highlights