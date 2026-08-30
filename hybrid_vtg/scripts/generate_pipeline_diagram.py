#!/usr/bin/env python3
"""
Generate publication-quality overview diagram for ScoutTG (Stage 1-4).
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

def generate_pipeline_diagram(output_paths):
    plt.rcParams.update({
        'font.size': 11,
        'font.family': 'sans-serif',
        'figure.dpi': 300,
    })

    fig, ax = plt.subplots(figsize=(11.0, 5.4))
    ax.set_xlim(0, 1100)
    ax.set_ylim(0, 540)
    ax.axis('off')

    def draw_box(x, y, w, h, title, subtitle, box_color, border_color, title_color='#111827', subtitle_color='#374151'):
        box = patches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0,rounding_size=12",
            facecolor=box_color,
            edgecolor=border_color,
            linewidth=1.5,
            zorder=2
        )
        ax.add_patch(box)
        
        ax.text(x + w / 2, y + h / 2 + 13, title, ha='center', va='center',
                fontsize=12, fontweight='bold', color=title_color, zorder=3)
        ax.text(x + w / 2, y + h / 2 - 14, subtitle, ha='center', va='center',
                fontsize=9.5, color=subtitle_color, zorder=3)

    # Box dimensions
    bw, bh = 420, 85

    # Row 1: y = 400
    # Box 1: Raw Video
    draw_box(50, 400, bw, bh,
             "Raw Video", "0 to T seconds (untrimmed)",
             box_color='#f8fafc', border_color='#cbd5e1',
             title_color='#0f172a', subtitle_color='#475569')

    # Box 2: Stage 1 - Temporal Scout
    draw_box(630, 400, bw, bh,
             "Stage 1 — Temporal Scout", "Frozen 1.04B Scout at 1.0 FPS scan",
             box_color='#ecfdf5', border_color='#10b981',
             title_color='#065f46', subtitle_color='#047857')

    # Row 2: y = 230
    # Box 3: Stage 2 - Proposal Extraction
    draw_box(50, 230, bw, bh,
             "Stage 2 — Proposal Extraction", "Z-score normalization & energy scoring",
             box_color='#ecfdf5', border_color='#10b981',
             title_color='#065f46', subtitle_color='#047857')

    # Box 4: Stage 3 - Window Planning
    draw_box(630, 230, bw, bh,
             "Stage 3 — Window Planning", "Adaptive margins & continuity merging",
             box_color='#eef2ff', border_color='#6366f1',
             title_color='#3730a3', subtitle_color='#4338ca')

    # Row 3: y = 60
    # Box 5: Stage 4 - Dense Grounding
    draw_box(50, 60, bw, bh,
             "Stage 4 — Dense Grounding", "TimeLens-8B (64 to 256 frames)",
             box_color='#fff7ed', border_color='#f97316',
             title_color='#9a3412', subtitle_color='#c2410c')

    # Box 6: Multi-Span Predictions
    draw_box(630, 60, bw, bh,
             "Multi-Span Predictions", "Schema-delimited JSON timestamps",
             box_color='#f8fafc', border_color='#cbd5e1',
             title_color='#0f172a', subtitle_color='#475569')

    # Connecting Arrows
    def draw_arrow(x1, y1, x2, y2, color='#64748b', lw=1.8):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                    mutation_scale=15, shrinkA=0, shrinkB=0),
                    zorder=1)

    # 1 -> 2 (Horizontal)
    draw_arrow(470, 442.5, 630, 442.5)

    # 2 -> 3 (S-curve path)
    ax.plot([840, 840, 260, 260], [400, 355, 355, 315], color='#64748b', lw=1.8, zorder=1)
    draw_arrow(260, 318, 260, 315)

    # 3 -> 4 (Horizontal)
    draw_arrow(470, 272.5, 630, 272.5)

    # 4 -> 5 (S-curve path at y=168)
    ax.plot([840, 840, 260, 260], [230, 168, 168, 145], color='#64748b', lw=1.8, zorder=1)
    draw_arrow(260, 148, 260, 145)

    # Text placed neatly above the connecting wire
    ax.text(550, 188, "50%–85% background removed", ha='center', va='center',
            fontsize=10.5, fontweight='bold', color='#334155', zorder=3)

    # 5 -> 6 (Horizontal)
    draw_arrow(470, 102.5, 630, 102.5)

    for p in output_paths:
        os.makedirs(os.path.dirname(p), exist_ok=True)
        fig.savefig(p, dpi=300, bbox_inches='tight')
        print(f"Saved: {p}")
    plt.close(fig)

if __name__ == '__main__':
    generate_pipeline_diagram([
        '/home/dat/read_papers/hybrid_vtg/paper/images/pipeline_large.png',
        '/home/dat/read_papers/hybrid_vtg/docs/figures/pipeline_large.png'
    ])
