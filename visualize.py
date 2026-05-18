"""
TFT-MPIR 实时训练曲线可视化
训练过程中每个 epoch 结束后刷新图像，训练完成后保存为 PNG。

支持两种模式：
  - 有显示器环境（本地）：弹出实时刷新的窗口
  - 无显示器环境（服务器/Jupyter）：每个 epoch 覆盖保存 training_curve.png
    直接用文件管理器刷新即可查看进度
"""

import os
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np


def _detect_display() -> bool:
    if os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY'):
        return True
    try:
        import tkinter
        tkinter.Tk().destroy()
        return True
    except Exception:
        return False


class TrainingVisualizer:
    """
    用法
    ----
    viz = TrainingVisualizer(save_dir='./checkpoints')
    viz.update(epoch=1, train_loss=0.08, val_loss=0.26, lr=0.01)  # 每 epoch 调用
    viz.save_final()   # 训练结束后调用
    """

    def __init__(self, save_dir: str = './checkpoints'):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        self.epochs:       list = []
        self.train_losses: list = []
        self.val_losses:   list = []
        self.lrs:          list = []
        self.best_epoch:   int   = 0
        self.best_val:     float = float('inf')

        has_display = _detect_display()
        matplotlib.use('TkAgg' if has_display else 'Agg')

        self.fig = plt.figure(figsize=(13, 5), facecolor='#0f1117')
        gs = gridspec.GridSpec(1, 2, figure=self.fig, wspace=0.35,
                               left=0.08, right=0.97, top=0.88, bottom=0.13)
        self.ax_loss = self.fig.add_subplot(gs[0])
        self.ax_lr   = self.fig.add_subplot(gs[1])
        self._style_axes()
        self.fig.suptitle('TFT-MPIR 训练监控', color='#e0e0e0',
                          fontsize=14, fontweight='bold', y=0.97)

        if has_display:
            plt.ion()
            plt.show(block=False)

    def _style_axes(self):
        for ax in [self.ax_loss, self.ax_lr]:
            ax.set_facecolor('#1a1d27')
            ax.tick_params(colors='#9ca3af', labelsize=9)
            for spine in ['bottom', 'left']:
                ax.spines[spine].set_color('#374151')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.grid(True, color='#1f2937', linewidth=0.8, linestyle='--')
            ax.yaxis.label.set_color('#9ca3af')
            ax.xaxis.label.set_color('#9ca3af')
            ax.title.set_color('#e0e0e0')

    def update(self, epoch: int, train_loss: float, val_loss: float, lr: float):
        self.epochs.append(epoch)
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.lrs.append(lr)

        if val_loss < self.best_val:
            self.best_val   = val_loss
            self.best_epoch = epoch

        self._draw()
        self.fig.savefig(
            os.path.join(self.save_dir, 'training_curve.png'),
            dpi=120, bbox_inches='tight', facecolor=self.fig.get_facecolor()
        )
        try:
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
        except Exception:
            pass

    def _draw(self):
        ep = self.epochs

        # ── Loss 曲线 ──────────────────────────
        ax = self.ax_loss
        ax.cla()
        self._style_axes()

        ax.plot(ep, self.train_losses, color='#60a5fa', linewidth=1.8,
                label='Train Loss', zorder=3)
        ax.plot(ep, self.val_losses,   color='#f472b6', linewidth=1.8,
                label='Val Loss',   zorder=3)
        ax.fill_between(ep, self.train_losses, self.val_losses,
                        alpha=0.08, color='#a78bfa')

        # 最优点标注
        if self.best_epoch in ep:
            idx = ep.index(self.best_epoch)
            ax.axvline(self.best_epoch, color='#34d399', linewidth=1,
                       linestyle='--', alpha=0.7, zorder=2)
            ax.scatter([self.best_epoch], [self.val_losses[idx]],
                       color='#34d399', s=60, zorder=5)
            ax.annotate(
                f'best\n{self.best_val:.4f}',
                xy=(self.best_epoch, self.val_losses[idx]),
                xytext=(6, -18), textcoords='offset points',
                color='#34d399', fontsize=8,
            )

        # 最新值标注
        ax.annotate(f'{self.train_losses[-1]:.4f}',
                    xy=(ep[-1], self.train_losses[-1]),
                    xytext=(4, 4), textcoords='offset points',
                    color='#60a5fa', fontsize=8)
        ax.annotate(f'{self.val_losses[-1]:.4f}',
                    xy=(ep[-1], self.val_losses[-1]),
                    xytext=(4, -12), textcoords='offset points',
                    color='#f472b6', fontsize=8)

        ax.set_title('Loss 曲线', fontsize=11, pad=8)
        ax.set_xlabel('Epoch', fontsize=9)
        ax.set_ylabel('Quantile Loss', fontsize=9)
        ax.legend(framealpha=0.2, labelcolor='white', fontsize=9,
                  facecolor='#1a1d27', edgecolor='#374151')

        # ── 学习率曲线 ─────────────────────────
        ax2 = self.ax_lr
        ax2.cla()
        self._style_axes()

        ax2.plot(ep, self.lrs, color='#fbbf24', linewidth=1.8, zorder=3)
        ax2.fill_between(ep, self.lrs, alpha=0.15, color='#fbbf24')

        for i in range(1, len(self.lrs)):
            if self.lrs[i] < self.lrs[i - 1]:
                ax2.axvline(ep[i], color='#fb923c', linewidth=1,
                            linestyle=':', alpha=0.8)
                ax2.annotate(f'LR↓\n{self.lrs[i]:.1e}',
                             xy=(ep[i], self.lrs[i]),
                             xytext=(4, 4), textcoords='offset points',
                             color='#fb923c', fontsize=8)

        ax2.set_yscale('log')
        ax2.set_title('学习率变化', fontsize=11, pad=8)
        ax2.set_xlabel('Epoch', fontsize=9)
        ax2.set_ylabel('Learning Rate (log)', fontsize=9)

        # 底部状态栏
        status = (f"Epoch {ep[-1]}  |  "
                  f"Train {self.train_losses[-1]:.4f}  |  "
                  f"Val {self.val_losses[-1]:.4f}  |  "
                  f"Best Val {self.best_val:.4f} @ Ep{self.best_epoch}  |  "
                  f"LR {self.lrs[-1]:.2e}")
        self.fig.texts.clear()
        self.fig.text(0.5, 0.01, status, ha='center', va='bottom',
                      color='#6b7280', fontsize=8.5, fontfamily='monospace')
        self.fig.suptitle('TFT-MPIR 训练监控', color='#e0e0e0',
                          fontsize=14, fontweight='bold', y=0.97)

    def save_final(self):
        out = os.path.join(self.save_dir, 'training_curve_final.png')
        self.fig.savefig(out, dpi=150, bbox_inches='tight',
                         facecolor=self.fig.get_facecolor())
        print(f"训练曲线已保存: {out}")

    def close(self):
        plt.close(self.fig)


# ── 离线重绘（从 checkpoint 或 metrics.json）──────────────────────

def plot_from_checkpoint(ckpt_path: str, save_dir: str = None):
    """
    训练已结束时，从保存的文件重新绘制曲线。

    示例：
        python visualize.py --ckpt ./checkpoints/last_checkpoint.pt
        python visualize.py --ckpt ./checkpoints/metrics.json
    """
    import json

    if ckpt_path.endswith('.pt'):
        import torch
        ckpt    = torch.load(ckpt_path, map_location='cpu')
        history = ckpt['history']
        lrs     = history.get('lr', [None] * len(history['train_loss']))
    else:
        with open(ckpt_path) as f:
            data = json.load(f)
        history = data['history']
        lrs     = history.get('lr', [None] * len(history['train_loss']))

    save_dir = save_dir or os.path.dirname(os.path.abspath(ckpt_path))
    viz = TrainingVisualizer(save_dir=save_dir)

    for i, (tl, vl, lr) in enumerate(
        zip(history['train_loss'], history['val_loss'], lrs), 1
    ):
        viz.update(epoch=i, train_loss=tl, val_loss=vl, lr=lr or 0.01)

    viz.save_final()
    print(f"共 {len(history['train_loss'])} 个 epoch 的曲线绘制完成。")
    return viz


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description='离线重绘训练曲线')
    p.add_argument('--ckpt', type=str, required=True,
                   help='last_checkpoint.pt 或 metrics.json 路径')
    p.add_argument('--save_dir', type=str, default='')
    args = p.parse_args()
    plot_from_checkpoint(args.ckpt, args.save_dir or None)